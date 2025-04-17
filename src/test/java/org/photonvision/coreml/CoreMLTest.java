package org.photonvision.coreml;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import java.io.File;
import java.nio.file.Paths;

public class CoreMLTest {
    
    @Test
    public void testBasicJNI() {
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
        
        // Test create
        long ptr = CoreMLJNI.create("test.model", 1, CoreMLJNI.ModelVersion.YOLO_V5.ordinal(), 1);
        assertEquals(0, ptr);
        
        // Test setCoreMask
        int ret = CoreMLJNI.setCoreMask(ptr, 1);
        assertEquals(0, ret);
        
        // Test destroy
        CoreMLJNI.destroy(ptr);
    }
} 