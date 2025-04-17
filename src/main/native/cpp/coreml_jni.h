#ifndef COREML_JNI_H_
#define COREML_JNI_H_

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_org_photonvision_coreml_CoreMLJNI_create
  (JNIEnv *, jclass, jstring, jint, jint, jint);

JNIEXPORT jint JNICALL Java_org_photonvision_coreml_CoreMLJNI_setCoreMask
  (JNIEnv *, jclass, jlong, jint);

JNIEXPORT void JNICALL Java_org_photonvision_coreml_CoreMLJNI_destroy
  (JNIEnv *, jclass, jlong);

JNIEXPORT jobjectArray JNICALL Java_org_photonvision_coreml_CoreMLJNI_detect
  (JNIEnv *, jclass, jlong, jlong, jdouble, jdouble);

#ifdef __cplusplus
}
#endif

#endif // COREML_JNI_H_ 