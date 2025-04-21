package org.photonvision.coreml;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import static org.junit.jupiter.api.Assertions.*;

public class CoreMLDetectionTest {
    
    @BeforeAll
    public static void setup() {
        CoreMLTestUtils.initializeLibraries();
    }

    @Test
    public void testCoralDetection() {
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL);
        assertNotEquals(0, ptr, "Model creation should return valid pointer");

        // Test coral detection
        Mat image = CoreMLTestUtils.loadTestImage("coral.jpeg");
        assertNotNull(image, "Test image should be loaded");
        assertFalse(image.empty(), "Test image should not be empty");

        var results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, 0.5);
        assertNotNull(results, "Detection results should not be null");
        assertTrue(results.length > 0, "Should detect coral in the image");

        // Verify detection results
        for (var result : results) {
            assertTrue(result.conf > 0 && result.conf <= 1.0, "Confidence should be between 0 and 1");
            assertTrue(result.class_id >= 0, "Class ID should be non-negative");
            assertTrue(result.rect.width > 0 && result.rect.height > 0, "Detection box should have positive dimensions");
            
            // Draw detection results for debugging
            Scalar boxColor = new Scalar(255, 0, 0);
            Point pt1 = new Point(result.rect.x, result.rect.y);
            Point pt2 = new Point(result.rect.x + result.rect.width, result.rect.y + result.rect.height);
            Imgproc.rectangle(image, pt1, pt2, boxColor, 2);
            
            String label = String.format("Coral: %.2f", result.conf);
            Point labelOrg = new Point(result.rect.x, result.rect.y - 10);
            Imgproc.putText(image, label, labelOrg, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 2);
        }
        
        CoreMLTestUtils.saveDebugImage(image, "coral_detection_result.jpg");
        CoreMLJNI.destroy(ptr);
    }

    @Test
    public void testAlgaeDetection() {
        String modelPath = CoreMLTestUtils.loadTestModel("algae-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL);
        assertNotEquals(0, ptr, "Model creation should return valid pointer");

        // Test algae detection
        Mat image = CoreMLTestUtils.loadTestImage("algae.jpeg");
        assertNotNull(image, "Test image should be loaded");
        assertFalse(image.empty(), "Test image should not be empty");

        var results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, 0.5);
        assertNotNull(results, "Detection results should not be null");
        assertTrue(results.length > 0, "Should detect algae in the image");

        // Verify detection results
        for (var result : results) {
            assertTrue(result.conf > 0 && result.conf <= 1.0, "Confidence should be between 0 and 1");
            assertTrue(result.class_id >= 0, "Class ID should be non-negative");
            assertTrue(result.rect.width > 0 && result.rect.height > 0, "Detection box should have positive dimensions");
            
            // Draw detection results for debugging
            Scalar boxColor = new Scalar(255, 0, 0);
            Point pt1 = new Point(result.rect.x, result.rect.y);
            Point pt2 = new Point(result.rect.x + result.rect.width, result.rect.y + result.rect.height);
            Imgproc.rectangle(image, pt1, pt2, boxColor, 2);
            
            String label = String.format("Algae: %.2f", result.conf);
            Point labelOrg = new Point(result.rect.x, result.rect.y - 10);
            Imgproc.putText(image, label, labelOrg, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 2);
        }
        
        CoreMLTestUtils.saveDebugImage(image, "algae_detection_result.jpg");
        CoreMLJNI.destroy(ptr);
    }

    @Test
    public void testDetectionPerformance() {
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL);
        Mat image = CoreMLTestUtils.loadTestImage("coral.jpeg");
        
        // Warm up
        CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, 0.5);
        
        // Test detection performance
        int numIterations = 10;
        long startTime = System.nanoTime();
        
        for (int i = 0; i < numIterations; i++) {
            var results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, 0.5);
            assertNotNull(results, "Detection results should not be null");
        }
        
        long endTime = System.nanoTime();
        double avgTimeMs = (endTime - startTime) / (numIterations * 1_000_000.0);
        
        System.out.println("Average detection time: " + avgTimeMs + " ms");
        assertTrue(avgTimeMs < 1000, "Average detection time should be less than 1000ms");
        
        CoreMLJNI.destroy(ptr);
    }

    @Test
    public void testDifferentConfidenceThresholds() {
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL);
        Mat image = CoreMLTestUtils.loadTestImage("coral.jpeg");
        
        double[] confidenceThresholds = {0.1, 0.3, 0.5, 0.7, 0.9};
        int[] detectionCounts = new int[confidenceThresholds.length];
        
        for (int i = 0; i < confidenceThresholds.length; i++) {
            var results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, confidenceThresholds[i]);
            detectionCounts[i] = results.length;
            
            // Higher confidence threshold should result in fewer detections
            if (i > 0) {
                assertTrue(detectionCounts[i] <= detectionCounts[i-1], 
                    "Higher confidence threshold should result in fewer or equal detections");
            }
        }
        
        CoreMLJNI.destroy(ptr);
    }
} 