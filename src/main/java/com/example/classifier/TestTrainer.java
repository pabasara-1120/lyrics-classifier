package com.example.classifier;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TestTrainer {
    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder()
                .appName("TestTrainer")
                .master("local[*]")
                .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
                .config("spark.kryo.registrationRequired", "false")
                .getOrCreate();

        Dataset<Row> data = spark.read()
                .option("header", "true")
                .csv("src/main/resources/data/tcc_ceds_music.csv")
                .select("lyrics", "genre")
                .na().drop();

        // pipeline steps as in your service...

        spark.stop();
    }
}
