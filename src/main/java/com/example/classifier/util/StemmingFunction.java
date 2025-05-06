package com.example.classifier.util;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.tartarus.snowball.ext.EnglishStemmer;

public class StemmingFunction implements MapFunction<Row, Row> {

    private final EnglishStemmer stemmer = new EnglishStemmer();

    @Override
    public Row call(Row input) throws Exception {
        String word = input.getAs("filtered_word"); // Make sure this matches your schema
        if (word == null) {
            return input; // Skip or handle null values
        }

        stemmer.setCurrent(word);
        stemmer.stem();
        String stemmedWord = stemmer.getCurrent();

        return RowFactory.create(
                input.getAs("id"),               // assuming "id" is your identifier column
                input.getAs("row_number"),       // your row number
                input.getAs("label"),            // your label (genre index)
                stemmedWord                      // stemmed word as final column
        );
    }
}
