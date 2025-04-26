#include "coreml_jni.h"
#include "CoreMLDetector.h"
#include "jni.h"
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

    detectionResultClass = JClass(env, "org/atomstorm/coreml/CoreMLJNI$CoreMLResult");
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

JNIEXPORT jlong JNICALL Java_org_atomstorm_coreml_CoreMLJNI_create
  (JNIEnv *env, jclass, jstring modelPath, jint numClasses, jint modelVer, jint coreNum)
{
    if (modelPath == nullptr) {
        return 0; // Return NULL if modelPath is invalid
    }

    const char* modelPathStr = env->GetStringUTFChars(modelPath, nullptr);
    if (modelPathStr == nullptr) {
        return 0; // Return NULL if string conversion fails
    }

    std::string modelPathCpp(modelPathStr);
    env->ReleaseStringUTFChars(modelPath, modelPathStr);
    
    CoreMLContext* context = new (std::nothrow) CoreMLContext();
    if (context == nullptr) {
        return 0; // Return NULL if context allocation fails
    }

    try {
        context->detector = new CoreMLDetector(modelPathCpp);
        if (context->detector == nullptr) {
            delete context;
            return 0;
        }

        if (context->detector->setCoreMask(coreNum) != 0) {
            delete context->detector;
            delete context;
            return 0;
        }
        
        return reinterpret_cast<jlong>(context);
    } catch (...) {
        // Clean up if any exception occurs
        if (context != nullptr) {
            delete context->detector;
            delete context;
        }
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_org_atomstorm_coreml_CoreMLJNI_setCoreMask
  (JNIEnv *env, jclass, jlong ptr, jint desiredCore)
{
    if (ptr == 0) {
        return -1; // Invalid pointer
    }
    
    CoreMLContext* context = reinterpret_cast<CoreMLContext*>(ptr);
    if (context == nullptr || context->detector == nullptr) {
        return -1; // Invalid context or detector
    }
    
    return context->detector->setCoreMask(desiredCore);
}

JNIEXPORT void JNICALL Java_org_atomstorm_coreml_CoreMLJNI_destroy
  (JNIEnv *env, jclass, jlong ptr)
{
    if (ptr == 0) {
        return; // Invalid pointer, nothing to destroy
    }
    
    CoreMLContext* context = reinterpret_cast<CoreMLContext*>(ptr);
    if (context == nullptr) {
        return; // Invalid context
    }
    
    if (context->detector != nullptr) {
        delete context->detector;
    }
    delete context;
}

JNIEXPORT jobjectArray JNICALL Java_org_atomstorm_coreml_CoreMLJNI_detect
  (JNIEnv *env, jclass, jlong detectorPtr, jlong imagePtr, jdouble nmsThresh, jdouble boxThresh)
{   
    jobjectArray emptyArray = env->NewObjectArray(0, detectionResultClass, nullptr);

    if (detectorPtr == 0 || imagePtr == 0) {
        return emptyArray;
    }
    
    CoreMLContext* context = reinterpret_cast<CoreMLContext*>(detectorPtr);
    if (context == nullptr || context->detector == nullptr) {
        return emptyArray;
    }
    
    cv::Mat* image = reinterpret_cast<cv::Mat*>(imagePtr);
    if (image == nullptr) {
        return emptyArray;
    }

    if (nmsThresh < 0.0 || nmsThresh > 1.0) {
        return emptyArray;
    }

    if (boxThresh < 0.0 || boxThresh > 1.0) {
        return emptyArray;
    }
    
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