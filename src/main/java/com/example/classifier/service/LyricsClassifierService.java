package com.example.classifier.service;

import com.example.classifier.pipeline.LogisticRegressionPipeline;
import com.example.classifier.transformer.*;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.tartarus.snowball.ext.EnglishStemmer;
import org.apache.spark.sql.api.java.UDF1;

import java.io.IOException;
import java.util.*;

import static org.apache.spark.sql.functions.monotonically_increasing_id;


@Service
public class LyricsClassifierService {

    @Autowired
    private SparkSession spark;

    private CrossValidatorModel model;
    private StringIndexerModel labelIndexer;
    private IndexToString labelConverter;
    private String[] labels;


//    public void trainModel() {
//        Dataset<Row> rawData = spark.read()
//                .option("header", "true")
//                .csv("src/main/resources/data/tcc_ceds_music.csv")
//                .select("lyrics", "genre")
//                .na().drop();
//
//        labelIndexer = new StringIndexer()
//                .setInputCol("genre")
//                .setOutputCol("label")
//                .fit(rawData);
//
//        Tokenizer tokenizer = new Tokenizer()
//                .setInputCol("lyrics")
//                .setOutputCol("words");
//
//        StopWordsRemover remover = new StopWordsRemover()
//                .setInputCol("words")
//                .setOutputCol("filtered");
//
//        SQLTransformer toVerse = new SQLTransformer()
//                .setStatement("SELECT *, concat_ws(' ', filtered) AS verse FROM __THIS__");
//
//        Word2Vec word2Vec = new Word2Vec()
//                .setInputCol("filtered") // or "verse" if you use sentence grouping
//                .setOutputCol("features")
//                .setVectorSize(200)
//                .setMinCount(1);
//
//        LogisticRegression lr = new LogisticRegression()
//                .setMaxIter(100)
//                .setRegParam(0.01);
//
//        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
//                labelIndexer, tokenizer, remover, toVerse, word2Vec, lr
//        });
//
//        Dataset<Row>[] splits = rawData.randomSplit(new double[]{0.8, 0.2}, 1234);
//        Dataset<Row> training = splits[0];
//        Dataset<Row> test = splits[1];
//
//        model = pipeline.fit(training);
//
//        Dataset<Row> predictions = model.transform(test);
//
//        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
//                .setLabelCol("label")
//                .setPredictionCol("prediction")
//                .setMetricName("accuracy");
//
//        double accuracy = evaluator.evaluate(predictions);
//        System.out.println("Test accuracy: " + accuracy);
//
//        labelConverter = new IndexToString()
//                .setInputCol("prediction")
//                .setOutputCol("predictedGenre")
//                .setLabels(labelIndexer.labels());
//    }

    public void trainModel() throws IOException {
        Dataset<Row> raw = spark.read()
                .option("header", "true")
                .csv("src/main/resources/data/tcc_ceds_music.csv")
                .select("lyrics", "genre")
                .na().drop()
                .withColumn("id", monotonically_increasing_id());;


        this.labelIndexer = new StringIndexer()
                .setInputCol("genre")
                .setOutputCol("label")
                .fit(raw);
        Dataset<Row> indexed = labelIndexer.transform(raw);
        labels = labelIndexer.labels();

        SQLTransformer cleanser = new SQLTransformer()
                .setStatement("SELECT *, regexp_replace(trim(lyrics), '[^\\\\w\\\\s]', '') AS clean FROM __THIS__");

        Numerator numerator = new Numerator();

        Tokenizer tokenizer = new Tokenizer()
                .setInputCol("clean")
                .setOutputCol("words");

        StopWordsRemover stopWordsRemover = new StopWordsRemover()
                .setInputCol("words")
                .setOutputCol("filteredWords");

        Exploder exploder = new Exploder();


        spark.udf().register("stem", (UDF1<String, String>) word -> {
            if (word == null) return null;
            EnglishStemmer stemmer = new EnglishStemmer();
            stemmer.setCurrent(word);
            if (stemmer.stem()) {
                return stemmer.getCurrent();
            }
            return word;
        }, DataTypes.StringType);


        SQLTransformer stemmer = new SQLTransformer()
                .setStatement("SELECT *, stem(filteredWord) as stemmedWord FROM __THIS__");
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

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        CrossValidator cv = new CrossValidator()
                .setEstimator(pipeline)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(grid)
                .setNumFolds(5);

        model = cv.fit(indexed);

        Dataset<Row> predictions = model.transform(indexed);
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Accuracy: " + accuracy);

        model.write().overwrite().save("src/main/resources/spark-models/lyrics-lr-model");

    }


    public Map<String, Double> classifyLyrics(String lyrics) {

        List<Row> rows = Collections.singletonList(RowFactory.create(lyrics));
        StructType schema = new StructType(new StructField[]{
                new StructField("lyrics", DataTypes.StringType, false, Metadata.empty())
        });

        Dataset<Row> lyricsDF = spark.createDataFrame(rows, schema)
                .withColumn("id", functions.monotonically_increasing_id())  // Required for Numerator
                .withColumn("label", functions.lit(0.0)); // Dummy label (for compatibility)



        // Transform input
        Dataset<Row> predictions = model.transform(lyricsDF);
        Dataset<Row> labeledPredictions = labelConverter != null
                ? labelConverter.transform(predictions)
                : predictions;

        // Extract probability
        Row row = labeledPredictions.select("probability").first();
        Vector probabilities = row.getAs("probability");

        String[] labels = labelIndexer.labels();
        Map<String, Double> genreProbs = new LinkedHashMap<>();
        for (int i = 0; i < labels.length; i++) {
            genreProbs.put(labels[i], probabilities.apply(i));
        }

        return genreProbs;

    }

}
