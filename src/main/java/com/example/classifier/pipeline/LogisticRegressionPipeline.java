package com.example.classifier.pipeline;

import com.example.classifier.transformer.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.springframework.beans.factory.annotation.Autowired;

import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;



public class LogisticRegressionPipeline {

    @Autowired
    private SparkSession spark;

    private CrossValidatorModel model;
    private String[] labels;

    public void trainAndEvaluate() throws IOException {
        Dataset<Row> raw = spark.read()
                .option("header", "true")
                .csv("src/main/resources/data/tcc_ceds_music.csv")
                .select("lyrics", "genre")
                .na().drop();

        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("genre")
                .setOutputCol("label")
                .fit(raw);

        Dataset<Row> indexed = labelIndexer.transform(raw);
        labels = labelIndexer.labels();

        Cleanser cleanser = new Cleanser();
        Numerator numerator = new Numerator();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("clean")
                .setOutputCol("words");

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filtered_words");

        Exploder exploder = new Exploder();
        Stemmer stemmer = new Stemmer();
        Uniter uniter = new Uniter();
        Verser verser = new Verser();

        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("verse")
                .setOutputCol("features")
                .setMinCount(0);

        LogisticRegression lr = new LogisticRegression();

        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                cleanser, numerator, tokenizer, stopWordsRemover, exploder,
                stemmer, uniter, verser, word2Vec, lr
        });

        ParamMap[] grid = new ParamGridBuilder()
                .addGrid(verser.sentencesInVerse(), new int[]{4, 8})
                .addGrid(word2Vec.vectorSize(), new int[]{100, 200})
                .addGrid(lr.regParam(), new double[]{0.01})
                .addGrid(lr.maxIter(), new int[]{100, 200})
                .build();

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(new BinaryClassificationEvaluator())
                .setEstimatorParamMaps(grid)
                .setNumFolds(5);

        model = cv.fit(indexed);

        // 5. Evaluation
        Dataset<Row> predictions = model.transform(indexed);
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator();
        double auc = evaluator.evaluate(predictions);
        System.out.println("Area Under ROC: " + auc);

        // 6. Optional: Save model
        model.write().overwrite().save("src/main/resources/spark-models/lyrics-lr-model");
    }

    public Map<String, Double> classify(String lyrics) {
        if (model == null) throw new IllegalStateException("Model is not trained");

        Row row = RowFactory.create(lyrics);
        StructType schema = new StructType(new StructField[]{
                new StructField("lyrics", org.apache.spark.sql.types.DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> input = spark.createDataFrame(Collections.singletonList(row), schema);
        Dataset<Row> prediction = model.transform(input);

        Row predictionRow = prediction.select("probability").first();
        Vector probabilities = predictionRow.getAs("probability");

        Map<String, Double> genreProbs = new LinkedHashMap<>();
        for (int i = 0; i < labels.length; i++) {
            genreProbs.put(labels[i], probabilities.apply(i));
        }

        return genreProbs;
    }
}

