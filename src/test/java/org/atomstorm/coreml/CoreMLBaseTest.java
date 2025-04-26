package org.atomstorm.coreml;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import static org.junit.jupiter.api.Assertions.*;

public class CoreMLBaseTest {
    
    @BeforeAll
    public static void setup() {
        CoreMLTestUtils.initializeLibraries();
    }

    @Test
    public void testModelCreation() {
        // Test model creation with valid path
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL.ordinal());
        assertNotEquals(0, ptr, "Model creation should return valid pointer");
        CoreMLJNI.destroy(ptr);

        // Test model creation with invalid path
        String invalidPath = "invalid/path/model.mlmodel";
        ptr = CoreMLJNI.create(invalidPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL.ordinal());
        assertEquals(0, ptr, "Model creation with invalid path should return 0(NULL)");
    }

    @Test
    public void testCoreMaskSetting() {
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL.ordinal());
        
        // Test valid core mask settings
        int ret = CoreMLJNI.setCoreMask(ptr, CoreMLJNI.CoreMask.ALL);
        assertEquals(0, ret, "Setting ALL core mask should succeed");
        
        ret = CoreMLJNI.setCoreMask(ptr, CoreMLJNI.CoreMask.CPU_ONLY);
        assertEquals(0, ret, "Setting CPU_ONLY core mask should succeed");
        
        ret = CoreMLJNI.setCoreMask(ptr, CoreMLJNI.CoreMask.CPU_AND_GPU);
        assertEquals(0, ret, "Setting CPU_AND_GPU core mask should succeed");

        ret = CoreMLJNI.setCoreMask(ptr, CoreMLJNI.CoreMask.CPU_AND_NEURAL_ENGINE);
        assertEquals(0, ret, "Setting CPU_AND_NEURAL_ENGINE core mask should succeed");
        
        CoreMLJNI.destroy(ptr);
        
        // Test invalid pointer
        ret = CoreMLJNI.setCoreMask(0, CoreMLJNI.CoreMask.ALL);
        assertNotEquals(0, ret, "Setting core mask with invalid pointer should fail");
    }

    @Test
    public void testEmptyImageDetection() {
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL.ordinal());
        
        // Test with empty image
        Mat emptyImage = CoreMLTestUtils.loadTestImage("empty.png");
        assertNotNull(emptyImage, "Empty test image should be loaded");
        assertFalse(emptyImage.empty(), "Test image should not be empty");
        
        var results = CoreMLJNI.detect(ptr, emptyImage.getNativeObjAddr(), 0.5, 0.5);
        assertNotNull(results, "Detection results should not be null");
        assertEquals(0, results.length, "Empty image should have no detections");
        
        CoreMLJNI.destroy(ptr);
    }

    @Test
    public void testInvalidDetectionParameters() {
        String modelPath = CoreMLTestUtils.loadTestModel("coral-640-640-yolov11s.mlmodel");
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V11.ordinal(), CoreMLJNI.CoreMask.ALL.ordinal());
        
        Mat image = CoreMLTestUtils.loadTestImage("coral.jpeg");
        
        // Test invalid NMS threshold
        var results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), -1.0, 0.5);
        assertNotNull(results, "Detection with invalid NMS threshold should return empty array");
        assertEquals(0, results.length, "Detection with invalid NMS threshold should have no results");
        
        // Test invalid box threshold
        results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, -1.0);
        assertNotNull(results, "Detection with invalid box threshold should return empty array");
        assertEquals(0, results.length, "Detection with invalid box threshold should have no results");
        
        // Test invalid image pointer
        results = CoreMLJNI.detect(ptr, 0, 0.5, 0.5);
        assertNotNull(results, "Detection with invalid image should return empty array");
        assertEquals(0, results.length, "Detection with invalid image should have no results");
        
        CoreMLJNI.destroy(ptr);
    }
} 