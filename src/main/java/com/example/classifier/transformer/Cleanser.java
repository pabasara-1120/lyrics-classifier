package com.example.classifier.transformer;

import org.apache.spark.ml.Transformer;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.util.DefaultParamsWritable;
import org.apache.spark.ml.util.MLWriter;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.ml.util.DefaultParamsReader;
import org.apache.spark.ml.util.DefaultParamsWriter;
import org.apache.spark.ml.util.MLReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;

import java.io.IOException;
import java.util.UUID;

import static org.apache.spark.sql.functions.*;

public class Cleanser extends Transformer implements MLWritable, DefaultParamsWritable {

    private final String uid;

    private final Param<String> inputCol = new Param<>(this, "inputCol", "Input column name");
    private final Param<String> outputCol = new Param<>(this, "outputCol", "Output column name");

    public Cleanser() {
        this.uid = "Cleanser_" + UUID.randomUUID();
    }

    public Cleanser(String uid) {
        this.uid = uid;
    }

    public Cleanser setInputCol(String value) {
        set(inputCol, value);
        return this;
    }

    public Cleanser setOutputCol(String value) {
        set(outputCol, value);
        return this;
    }

    public String getInputCol() {
        return getOrDefault(inputCol);
    }

    public String getOutputCol() {
        return getOrDefault(outputCol);
    }

    @Override
    public Dataset<Row> transform(Dataset<?> dataset) {
        String input = getInputCol();
        String output = getOutputCol();

        Dataset<Row> transformed = dataset.withColumn(output,
                regexp_replace(trim(col(input)), "[^\\w\\s]", ""));
        transformed = transformed.withColumn(output,
                regexp_replace(col(output), "\\s{2,}", " "));
        return transformed;
    }

    @Override
    public StructType transformSchema(StructType schema) {
        return schema.add(getOutputCol(), org.apache.spark.sql.types.DataTypes.StringType, false);
    }

    @Override
    public String uid() {
        return this.uid;
    }

    @Override
    public Transformer copy(ParamMap extra) {
        return defaultCopy(extra);
    }

    @Override
    public MLWriter write() {
        return new DefaultParamsWriter(this);
    }

    public static MLReader<Cleanser> read() {
        return new DefaultParamsReader<>();
    }

    @Override
    public void save(String path) throws IOException {
        write().save(path);
    }
}
