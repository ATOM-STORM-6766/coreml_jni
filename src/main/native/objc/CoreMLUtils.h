#pragma once

// First include OpenCV headers to avoid macro conflicts
#include <opencv2/opencv.hpp>

// Then include Apple/Objective-C headers
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import "CoreMLDetector.h"

/**
 * Utility functions for CoreML detection operations.
 * This class provides helper methods for image processing, coordinate transformations,
 * and other common operations used in the CoreML detection pipeline.
 */
@interface CoreMLUtils : NSObject

/**
 * Structure to hold preprocessing parameters.
 */
typedef struct {
    NSInteger inputWidth;
    NSInteger inputHeight;
    float scaleFactor;
    float padWidth;
    float padHeight;
} PreprocessParams;

/**
 * Convert OpenCV Mat to CVPixelBuffer.
 * Uses the PixelBufferPool for efficient memory management.
 *
 * @param matimg The OpenCV Mat image to convert.
 * @return A CVPixelBuffer containing the image data, or nullptr if conversion failed.
 */
+ (CVPixelBufferRef)matToCVPixelBuffer:(const cv::Mat&)matimg;

/**
 * Preprocess an image for model input.
 * Resizes and pads the image to fit the model's input dimensions while preserving aspect ratio.
 *
 * @param image The input image to preprocess.
 * @param params Preprocessing parameters (will be updated with scaling and padding values).
 * @return The preprocessed image, or an empty Mat if preprocessing failed.
 */
+ (cv::Mat)preprocessImage:(const cv::Mat&)image params:(PreprocessParams*)params;

/**
 * Process detection results from the model.
 * Converts model output coordinates to original image coordinates.
 *
 * @param coords Pointer to the coordinates array from model output.
 * @param confs Pointer to the confidence array from model output.
 * @param imageWidth Original image width.
 * @param imageHeight Original image height.
 * @param numClasses Number of classes in the model.
 * @param params Preprocessing parameters used for the input image.
 * @return A DetectionResult structure with the processed detection.
 */
+ (DetectionResult)processDetectionResult:(float*)coords
                               confidence:(float*)confs
                              imageWidth:(int)imageWidth
                             imageHeight:(int)imageHeight
                              numClasses:(NSInteger)numClasses
                                  params:(PreprocessParams)params;

/**
 * Check if an image is valid for processing.
 *
 * @param image The image to validate.
 * @return YES if the image is valid, NO otherwise.
 */
+ (BOOL)validateImage:(const cv::Mat&)image;

/**
 * Get model input dimensions from a CoreML model.
 *
 * @param model The CoreML model.
 * @param params Preprocessing parameters to update with the model dimensions.
 * @return YES if dimensions were successfully retrieved, NO otherwise.
 */
+ (BOOL)getModelInputDimensions:(MLModel*)model params:(PreprocessParams*)params;

@end
