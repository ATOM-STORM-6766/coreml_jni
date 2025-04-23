package org.photonvision.coreml;

import static org.junit.jupiter.api.Assertions.*;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.opencv.core.*;
import org.photonvision.coreml.CoreMLJNI.CoreMLResult;

public class CoreMLThreadSafetyTest extends CoreMLBaseTest {
    private static final int NUM_THREADS = 4;
    private static final int NUM_ITERATIONS = 100;
    private static final double NMS_THRESH = 0.45;
    private static final double BOX_THRESH = 0.25;

    @Test
    public void testConcurrentDetection() throws InterruptedException, ExecutionException {
        // Load test image
        Mat image = CoreMLTestUtils.loadTestImage("coral.jpeg");
        assertNotNull(image, "Failed to load test image");
        
        // Create detector
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL);
        assertNotEquals(0, ptr, "Model creation should return valid pointer");
        
        try {
            // Create thread pool
            ExecutorService executor = Executors.newFixedThreadPool(NUM_THREADS);
            List<Future<CoreMLResult[]>> futures = new ArrayList<>();
            AtomicInteger successCount = new AtomicInteger(0);
            AtomicInteger errorCount = new AtomicInteger(0);

            // Submit detection tasks
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                futures.add(executor.submit(() -> {
                    try {
                        // Each thread gets its own copy of the image
                        Mat threadImage = image.clone();
                        CoreMLResult[] results = CoreMLJNI.detect(ptr, threadImage.getNativeObjAddr(), NMS_THRESH, BOX_THRESH);
                        successCount.incrementAndGet();
                        return results;
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        throw e;
                    }
                }));
            }

            // Wait for all tasks to complete
            executor.shutdown();
            assertTrue(executor.awaitTermination(60, TimeUnit.SECONDS), "Test timed out");

            // Verify results
            assertEquals(NUM_ITERATIONS, successCount.get(), "Not all detections completed successfully");
            assertEquals(0, errorCount.get(), "Some detections failed");

            // Check that all results are valid
            for (Future<CoreMLResult[]> future : futures) {
                CoreMLResult[] results = future.get();
                assertNotNull(results, "Detection results should not be null");
                // Verify that results are consistent
                for (CoreMLResult result : results) {
                    assertTrue(result.rect.x >= 0 && result.rect.x <= image.width(), "Invalid x1 coordinate");
                    assertTrue(result.rect.y >= 0 && result.rect.y <= image.height(), "Invalid y1 coordinate");
                    assertTrue(result.rect.x + result.rect.width <= image.width(), "Invalid x2 coordinate");
                    assertTrue(result.rect.y + result.rect.height <= image.height(), "Invalid y2 coordinate");
                    assertTrue(result.conf >= 0 && result.conf <= 1, "Invalid confidence");
                    assertTrue(result.class_id >= 0, "Invalid class ID");
                }
            }
        } finally {
            CoreMLJNI.destroy(ptr);
        }
    }

    @Test
    public void testStressDetection() throws InterruptedException {
        // Load test image
        Mat image = CoreMLTestUtils.loadTestImage("coral.jpeg");
        assertNotNull(image, "Failed to load test image");

        // Create detector
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL);
        assertNotEquals(0, ptr, "Model creation should return valid pointer");
        
        try {
            // 使用更合理的线程数，接近实际使用场景
            int numStressThreads = Runtime.getRuntime().availableProcessors();
            ExecutorService executor = Executors.newFixedThreadPool(numStressThreads);
            CountDownLatch latch = new CountDownLatch(numStressThreads);
            AtomicInteger successCount = new AtomicInteger(0);
            AtomicInteger errorCount = new AtomicInteger(0);
            AtomicLong totalProcessingTime = new AtomicLong(0);
            AtomicInteger totalDetections = new AtomicInteger(0);

            // 预热：先运行几次让 JIT 编译器优化代码
            for (int i = 0; i < 5; i++) {
                CoreMLJNI.detect(ptr, image.getNativeObjAddr(), NMS_THRESH, BOX_THRESH);
            }

            // Submit stress test tasks
            for (int i = 0; i < numStressThreads; i++) {
                executor.submit(() -> {
                    try {
                        // 每个线程运行多次检测
                        for (int j = 0; j < 20; j++) {
                            // 只测量核心检测时间
                            long startTime = System.nanoTime();
                            CoreMLResult[] results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), NMS_THRESH, BOX_THRESH);
                            long endTime = System.nanoTime();
                            
                            totalProcessingTime.addAndGet(endTime - startTime);
                            totalDetections.incrementAndGet();
                            
                            assertNotNull(results, "Detection results should not be null");
                        }
                        successCount.incrementAndGet();
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                    } finally {
                        latch.countDown();
                    }
                });
            }

            // Wait for all tasks to complete
            assertTrue(latch.await(60, TimeUnit.SECONDS), "Stress test timed out");
            executor.shutdown();

            // Calculate performance metrics
            double avgProcessingTimeMs = totalProcessingTime.get() / (double)totalDetections.get() / 1_000_000.0;
            double fps = 1000.0 / avgProcessingTimeMs;
            
            System.out.println("\n=== Stress Test Performance Metrics ===");
            System.out.printf("Number of threads: %d\n", numStressThreads);
            System.out.printf("Total detections: %d\n", totalDetections.get());
            System.out.printf("Average processing time: %.2f ms\n", avgProcessingTimeMs);
            System.out.printf("Average FPS: %.2f\n", fps);
            System.out.println("====================================\n");

            // Verify results
            assertEquals(numStressThreads, successCount.get(), "Not all stress test threads completed successfully");
            assertEquals(0, errorCount.get(), "Some stress test threads failed");
        } finally {
            CoreMLJNI.destroy(ptr);
        }
    }
} 