package com.example.classifier.config;

import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SparkConfig {

    @Bean
    public SparkSession sparkSession() {
        SparkConf conf = new SparkConf()
                .setAppName("LyricsClassifier")
                .setMaster("local[*]")
                .set("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
                .set("spark.kryo.registrationRequired", "false")
                .set("spark.kryo.unsafe", "false")
                .set("spark.serializer.objectStreamReset", "100")
                .set("spark.ui.enabled", "false")
                .set("spark.driver.memory", "2g");


        return SparkSession.builder()
                .config(conf)
                .getOrCreate();
    }
}
