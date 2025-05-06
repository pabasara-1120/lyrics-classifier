package com.example.classifier.transformer;

import java.io.IOException;
import java.util.UUID;

import com.example.classifier.enums.Column;
import com.example.classifier.util.StemmingFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.MapPartitionsFunction;
import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.encoders.RowEncoder;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import java.util.List;
import java.util.ArrayList;

public class Stemmer extends Transformer implements MLWritable {

    private String uid;

    public Stemmer(String uid) {
        this.uid = uid;
    }

    public Stemmer() {
        this.uid = "CustomStemmer" + "_" + UUID.randomUUID().toString();
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        StructType outputSchema = this.transformSchema(dataset.schema());
        Encoder<Row> rowEncoder = RowEncoder.encoderFor(outputSchema);

        return ((Dataset<Row>) dataset).mapPartitions(
                (MapPartitionsFunction<Row, Row>) iterator -> {
                    StemmingFunction stemmingFunction = new StemmingFunction();
                    List<Row> outputRows = new ArrayList<>();

                    while (iterator.hasNext()) {
                        Row inputRow = iterator.next();
                        outputRows.add(stemmingFunction.call(inputRow));
                    }

                    return outputRows.iterator();
                },
                rowEncoder
        );
    }


    @Override
    public StructType transformSchema(StructType schema) {
        return new StructType(new StructField[]{
                Column.ID.getStructType(),
                Column.ROW_NUMBER.getStructType(),
                Column.LABEL.getStructType(),
                Column.STEMMED_WORD.getStructType()
        });
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return super.defaultCopy(extra);
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }

    public static MLReader<Stemmer> read() {
        return new DefaultParamsReader<>();
    }

}

