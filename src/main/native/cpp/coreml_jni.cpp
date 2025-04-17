#include "coreml_jni.h"
#include <cstdio>
#include <opencv2/opencv.hpp>
// 定义一些基本的 OpenCV 变量和函数
cv::Mat img;
cv::Size size(640, 480);

void testOpenCV() {
    // 创建一个空白图像
    img = cv::Mat::zeros(size, CV_8UC3);
    
    printf("成功创建 OpenCV 测试图像\n");
}

extern "C" {

JNIEXPORT jlong JNICALL Java_org_photonvision_coreml_CoreMLJNI_create
  (JNIEnv *env, jclass, jstring modelPath, jint numClasses, jint modelVer, jint coreNum)
{
    testOpenCV();
    printf("CoreMLJNI: create called\n");
    return 0;
}

JNIEXPORT jint JNICALL Java_org_photonvision_coreml_CoreMLJNI_setCoreMask
  (JNIEnv *env, jclass, jlong ptr, jint desiredCore)
{
    printf("CoreMLJNI: setCoreMask called\n");
    return 0;
}

JNIEXPORT void JNICALL Java_org_photonvision_coreml_CoreMLJNI_destroy
  (JNIEnv *env, jclass, jlong ptr)
{
    printf("CoreMLJNI: destroy called\n");
}

JNIEXPORT jobjectArray JNICALL Java_org_photonvision_coreml_CoreMLJNI_detect
  (JNIEnv *env, jclass, jlong detectorPtr, jlong imagePtr, jdouble nmsThresh, jdouble boxThresh)
{
    printf("CoreMLJNI: detect called\n");
    return nullptr;
}

} // extern "C" 