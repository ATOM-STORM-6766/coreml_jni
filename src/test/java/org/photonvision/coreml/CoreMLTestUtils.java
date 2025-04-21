package org.photonvision.coreml;

import edu.wpi.first.util.CombinedRuntimeLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class CoreMLTestUtils {
    private static boolean initialized = false;

    /**
     * Initialize OpenCV and CoreML JNI libraries
     */
    public static void initializeLibraries() {
        if (initialized) return;
        
        try {
            // Load OpenCV
            CombinedRuntimeLoader.loadLibraries(CoreMLTestUtils.class, Core.NATIVE_LIBRARY_NAME);
            
            // Load CoreML JNI library from resources
            loadCoreMLJNI();
            
            initialized = true;
        } catch (IOException e) {
            throw new RuntimeException("Failed to initialize libraries", e);
        }
    }

    /**
     * Load CoreML JNI library from resources
     */
    private static void loadCoreMLJNI() throws IOException {
        String libraryName = "coreml_jni";
        String nativeLibName = System.mapLibraryName(libraryName);

        String path = Paths.get(System.getProperty("user.dir"), "cmake_build", nativeLibName).toString();
        // Load library
        System.load(path);
    }

    /**
     * Load test image from resources
     * @param imageName Image name in test resources
     * @return OpenCV Mat object
     */
    public static Mat loadTestImage(String imageName) {
        String imagePath = getResourcePath("2025/" + imageName);
        return Imgcodecs.imread(imagePath);
    }

    /**
     * Load test model from resources
     * @param modelName Model name in test resources
     * @return Path to model file
     */
    public static String loadTestModel(String modelName) {
        return getResourcePath("2025/" + modelName);
    }

    /**
     * Get absolute path for a resource file
     * @param resourcePath Relative path in test resources
     * @return Absolute path to resource
     */
    private static String getResourcePath(String resourcePath) {
        return Paths.get(System.getProperty("user.dir"), "src", "test", "resources", resourcePath).toString();
    }

    /**
     * Save detection result image for debugging
     * @param image Image with detection results drawn
     * @param filename Output filename
     */
    public static void saveDebugImage(Mat image, String filename) {
        Path outputPath = Paths.get(System.getProperty("user.dir"), "build", "test-results", filename);
        try {
            Files.createDirectories(outputPath.getParent());
            Imgcodecs.imwrite(outputPath.toString(), image);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
} 