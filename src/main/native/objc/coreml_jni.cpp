#include "coreml_jni.h"
#include "CoreMLDetector.h"
#include "wpi_jni_common.h"

struct CoreMLContext {
    CoreMLDetector* detector;
};

static JClass detectionResultClass;

extern "C" {

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
    JNIEnv *env;
    if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }

    detectionResultClass = JClass(env, "org/photonvision/coreml/CoreMLJNI$CoreMLResult");
    if (!detectionResultClass) {
        std::printf("Couldn't find class!");
        return JNI_ERR;
    }

    return JNI_VERSION_1_6;
}

static jobject MakeJObject(JNIEnv *env, const DetectionResult &result) {
    jmethodID constructor = env->GetMethodID(detectionResultClass, "<init>", "(IIIIFI)V");
    return env->NewObject(detectionResultClass, constructor, 
        static_cast<jint>(result.x1),
        static_cast<jint>(result.y1),
        static_cast<jint>(result.x2),
        static_cast<jint>(result.y2),
        static_cast<jfloat>(result.confidence),
        static_cast<jint>(result.class_id)
    );
}

JNIEXPORT jlong JNICALL Java_org_photonvision_coreml_CoreMLJNI_create
  (JNIEnv *env, jclass, jstring modelPath, jint numClasses, jint modelVer, jint coreNum)
{
    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    std::string modelPathCpp(modelPathStr);
    env->ReleaseStringUTFChars(modelPath, modelPathStr);
    
    CoreMLContext* context = new CoreMLContext();
    context->detector = new CoreMLDetector(modelPathCpp);
    context->detector->setCoreMask(coreNum);
    
    return reinterpret_cast<jlong>(context);
}

JNIEXPORT jint JNICALL Java_org_photonvision_coreml_CoreMLJNI_setCoreMask
  (JNIEnv *env, jclass, jlong ptr, jint desiredCore)
{
    CoreMLContext* context = reinterpret_cast<CoreMLContext*>(ptr);
    return context->detector->setCoreMask(desiredCore);
}

JNIEXPORT void JNICALL Java_org_photonvision_coreml_CoreMLJNI_destroy
  (JNIEnv *env, jclass, jlong ptr)
{
    CoreMLContext* context = reinterpret_cast<CoreMLContext*>(ptr);
    delete context->detector;
    delete context;
}

JNIEXPORT jobjectArray JNICALL Java_org_photonvision_coreml_CoreMLJNI_detect
  (JNIEnv *env, jclass, jlong detectorPtr, jlong imagePtr, jdouble nmsThresh, jdouble boxThresh)
{
    CoreMLContext* context = reinterpret_cast<CoreMLContext*>(detectorPtr);
    cv::Mat* image = reinterpret_cast<cv::Mat*>(imagePtr);
    
    std::vector<DetectionResult> results = context->detector->detect(*image, nmsThresh, boxThresh);

    jobjectArray jarr = env->NewObjectArray(results.size(), detectionResultClass, nullptr);
    
    for (size_t i = 0; i < results.size(); i++) {
        jobject obj = MakeJObject(env, results[i]);
        env->SetObjectArrayElement(jarr, i, obj);
        env->DeleteLocalRef(obj);
    }

    return jarr;
}

} // extern "C" 