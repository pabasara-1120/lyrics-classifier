package com.example.classifier.controller;

import com.example.classifier.service.LyricsClassifierService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class LyricsClassifierController {

    @Autowired
    private LyricsClassifierService lyricsClassifierService;

    @PostMapping("/train")
    public ResponseEntity<String> train() throws IOException {
        lyricsClassifierService.trainModel();
        return ResponseEntity.ok("Model trained successfully.");
    }

    @PostMapping("/classify")
    public Map<String, Double> classifyLyrics(@RequestBody String lyrics){
        return lyricsClassifierService.classifyLyrics(lyrics);
    }



}
