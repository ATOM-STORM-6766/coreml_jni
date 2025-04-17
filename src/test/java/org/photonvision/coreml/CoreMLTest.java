package org.photonvision.coreml;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import edu.wpi.first.util.CombinedRuntimeLoader;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;

public class CoreMLTest {
    @Test
    public void testBasicJNI() {
        try {
            CombinedRuntimeLoader.loadLibraries(CoreMLTest.class, Core.NATIVE_LIBRARY_NAME);
        } catch (IOException e) {
            e.printStackTrace();
        }


        // 构建 native 库的路径
        String libPath = Paths.get(System.getProperty("user.dir"), "cmake_build", "libcoreml_jni.dylib").toString();
        File libFile = new File(libPath);
        
        // 检查库文件是否存在
        if (!libFile.exists()) {
            fail("Native library not found at: " + libPath + 
                "\nPlease run './gradlew build' to build the project");
        }
        
        // 加载 native 库
        System.load(libPath);

        String modelPath = Paths.get(System.getProperty("user.dir"), "src", "test", "resources", "coral_v1.mlmodel").toString();
        
        // Test create
        long ptr = CoreMLJNI.create(modelPath, 1, CoreMLJNI.ModelVersion.YOLO_V5.ordinal(), 1);
        assertNotNull(ptr);

        // Test setCoreMask
        int ret = CoreMLJNI.setCoreMask(ptr, 1);
        assertEquals(0, ret);

        // Test detect
        String imagePath = Paths.get(System.getProperty("user.dir"), "src", "test", "resources", "coral.jpeg").toString();
        File imageFile = new File(imagePath);
        Mat image = Imgcodecs.imread(imageFile.getAbsolutePath());
        var results = CoreMLJNI.detect(ptr, image.getNativeObjAddr(), 0.5, 0.5);
        assertNotNull(results);

        var firstResult = results[0];

        assertTrue(firstResult.conf > 0);
        assertEquals(0, firstResult.class_id);

        // 在图像上绘制检测框和结果
        Scalar boxColor = new Scalar(0, 255, 0); // 绿色
        Point pt1 = new Point(firstResult.rect.x, firstResult.rect.y);
        Point pt2 = new Point(firstResult.rect.x + firstResult.rect.width, 
                            firstResult.rect.y + firstResult.rect.height);
        Imgproc.rectangle(image, pt1, pt2, boxColor, 2);

        // 添加置信度标签
        String label = String.format("Conf: %.2f", firstResult.conf);
        Point labelOrg = new Point(firstResult.rect.x, firstResult.rect.y - 10);
        Imgproc.putText(image, label, labelOrg, 
                       Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 2);
        // 保存结果图片用于测试验证
        String outputPath = Paths.get(System.getProperty("user.dir"), "build", "test-results", 
                                    "detection_result.jpg").toString();
        new File(outputPath).getParentFile().mkdirs();
        Imgcodecs.imwrite(outputPath, image);
        System.out.println("检测结果图片已保存至: " + outputPath);
        // Test destroy
        CoreMLJNI.destroy(ptr);
    }
} 