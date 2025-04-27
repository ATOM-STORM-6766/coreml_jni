// First include OpenCV headers to avoid macro conflicts
#include <opencv2/opencv.hpp>

// Then include our headers
#import "CoreMLUtils.h"
#import "PixelBufferPool.h"

// Define Log Levels (consistent with CoreMLDetector.mm)
#ifndef CURRENT_LOG_LEVEL
#define CURRENT_LOG_LEVEL 2 // Default to INFO level
#endif

#define LOG_LEVEL_NONE  0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_PERF  3
#define LOG_LEVEL_DEBUG 4

// Conditional Logging Macros
#if CURRENT_LOG_LEVEL >= LOG_LEVEL_ERROR
#define LOG_ERROR(fmt, ...) NSLog((@"[ERROR][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_ERROR(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL >= LOG_LEVEL_INFO
#define LOG_INFO(fmt, ...) NSLog((@"[INFO][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_INFO(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...) NSLog((@"[DEBUG][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) do {} while(0)
#endif

@implementation CoreMLUtils

+ (CVPixelBufferRef)matToCVPixelBuffer:(const cv::Mat&)matimg {
    // Create a copy of the input Mat and convert to BGRA
    cv::Mat bgra;
    cv::cvtColor(matimg, bgra, cv::COLOR_BGR2BGRA);

    // Get a pixel buffer from the pool
    CVPixelBufferRef pixelBuffer = [[PixelBufferPool sharedPool] getPixelBufferWithWidth:bgra.cols height:bgra.rows];
    if (!pixelBuffer) {
        LOG_ERROR("Failed to get pixel buffer from pool");
        return NULL;
    }

    // Lock the buffer for writing
    CVReturn status = CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    if (status != kCVReturnSuccess) {
        LOG_ERROR("Failed to lock pixel buffer: %d", status);
        [[PixelBufferPool sharedPool] returnPixelBuffer:pixelBuffer];
        return nullptr;
    }

    void* baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
    if (!baseAddress) {
        LOG_ERROR("Failed to get base address for pixel buffer");
        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
        [[PixelBufferPool sharedPool] returnPixelBuffer:pixelBuffer];
        return nullptr;
    }

    // Copy the image data
    memcpy(baseAddress, bgra.data, bgra.total() * bgra.elemSize());

    // Unlock the buffer
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);

    return pixelBuffer;
}

+ (cv::Mat)preprocessImage:(const cv::Mat&)image params:(PreprocessParams*)params {
    // Check if the input image is valid
    if (![self validateImage:image]) {
        return cv::Mat();
    }

    // If it is a grayscale image, convert it to RGB
    cv::Mat processedImage;
    if (image.channels() == 1) {
        cv::cvtColor(image, processedImage, cv::COLOR_GRAY2RGB);
    } else {
        processedImage = image.clone();
    }

    // Calculate scaling factor to fit within inputWidth x inputHeight while preserving aspect ratio
    params->scaleFactor = std::min((float)params->inputWidth / processedImage.cols,
                                  (float)params->inputHeight / processedImage.rows);

    // Calculate new dimensions after scaling
    int new_w = round(processedImage.cols * params->scaleFactor);
    int new_h = round(processedImage.rows * params->scaleFactor);

    // Ensure new dimensions are valid
    if (new_w <= 0 || new_h <= 0) {
        LOG_ERROR("Invalid scaled dimensions calculated: %d x %d", new_w, new_h);
        return cv::Mat();
    }

    // Calculate padding
    params->padWidth = (params->inputWidth - new_w) / 2.0f;
    params->padHeight = (params->inputHeight - new_h) / 2.0f;

    // Resize the image
    cv::Mat resized;
    cv::resize(processedImage, resized, cv::Size(new_w, new_h));

    // Create the final scaled/padded image buffer for the model
    cv::Mat image_scaled = cv::Mat::zeros(params->inputHeight, params->inputWidth, CV_8UC3);

    // Define the Region of Interest (ROI) in the scaled image where the resized image will be copied
    cv::Rect roi(round(params->padWidth), round(params->padHeight), new_w, new_h);

    // Ensure ROI is within the bounds of image_scaled
    if (roi.x >= 0 && roi.y >= 0 && roi.width > 0 && roi.height > 0 &&
        roi.x + roi.width <= image_scaled.cols && roi.y + roi.height <= image_scaled.rows) {
        // Copy the resized image into the center of the black canvas
        resized.copyTo(image_scaled(roi));
    } else {
        LOG_ERROR("Invalid ROI calculated for padding: x=%d, y=%d, w=%d, h=%d. Canvas size: %dx%d",
                 roi.x, roi.y, roi.width, roi.height, image_scaled.cols, image_scaled.rows);
        return cv::Mat();
    }

    return image_scaled;
}

+ (DetectionResult)processDetectionResult:(float*)coords
                               confidence:(float*)confs
                              imageWidth:(int)imageWidth
                             imageHeight:(int)imageHeight
                              numClasses:(NSInteger)numClasses
                                  params:(PreprocessParams)params {
    // Check if the input parameters are valid
    if (!coords || !confs || numClasses <= 0) {
        LOG_ERROR("Invalid input parameters");
        DetectionResult emptyResult;
        memset(&emptyResult, 0, sizeof(DetectionResult));
        return emptyResult;
    }

    // Find the class with the highest confidence
    float maxConf = confs[0];
    NSInteger objClass = 0;

    for (NSInteger i = 1; i < numClasses; i++) {
        if (confs[i] > maxConf) {
            maxConf = confs[i];
            objClass = i;
        }
    }

    // If the maximum confidence is less than the threshold, consider no object detected
    if (maxConf < 0.1f) {
        DetectionResult emptyResult;
        memset(&emptyResult, 0, sizeof(DetectionResult));
        return emptyResult;
    }

    // Convert normalized coordinates to absolute coordinates in the scaled/padded image space
    float abs_cx = coords[0] * params.inputWidth;
    float abs_cy = coords[1] * params.inputHeight;
    float abs_w = coords[2] * params.inputWidth;
    float abs_h = coords[3] * params.inputHeight;

    // Revert padding and scaling to get coordinates in the original image space
    float orig_cx = (abs_cx - params.padWidth) / params.scaleFactor;
    float orig_cy = (abs_cy - params.padHeight) / params.scaleFactor;
    float orig_w = abs_w / params.scaleFactor;
    float orig_h = abs_h / params.scaleFactor;

    // Calculate the four corners of the bounding box in original image coordinates
    float x1 = orig_cx - orig_w / 2.0f;
    float y1 = orig_cy - orig_h / 2.0f;
    float x2 = orig_cx + orig_w / 2.0f;
    float y2 = orig_cy + orig_h / 2.0f;

    // Ensure the coordinates are within the original image range
    x1 = std::max(0.0f, std::min(x1, (float)imageWidth));
    y1 = std::max(0.0f, std::min(y1, (float)imageHeight));
    x2 = std::max(0.0f, std::min(x2, (float)imageWidth));
    y2 = std::max(0.0f, std::min(y2, (float)imageHeight));

    // Create the detection result
    DetectionResult result;
    result.x1 = x1;
    result.y1 = y1;
    result.x2 = x2;
    result.y2 = y2;
    result.confidence = maxConf;
    result.class_id = objClass;

    return result;
}

+ (BOOL)validateImage:(const cv::Mat&)image {
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        LOG_ERROR("Invalid input image");
        return NO;
    }
    return YES;
}

+ (BOOL)getModelInputDimensions:(MLModel*)model params:(PreprocessParams*)params {
    if (!model || !params) {
        LOG_ERROR("Invalid model or params pointer");
        return NO;
    }

    MLModelDescription *modelDescription = model.modelDescription;
    NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptions = modelDescription.inputDescriptionsByName;
    MLFeatureDescription *imageInputDescription = inputDescriptions[@"image"];

    if (imageInputDescription && imageInputDescription.type == MLFeatureTypeImage) {
        MLImageConstraint *imageConstraint = imageInputDescription.imageConstraint;
        params->inputWidth = imageConstraint.pixelsWide;
        params->inputHeight = imageConstraint.pixelsHigh;

        if (params->inputWidth <= 0 || params->inputHeight <= 0) {
            LOG_ERROR("Invalid input dimensions retrieved from model: %ld x %ld",
                     params->inputWidth, params->inputHeight);
            return NO;
        }
        return YES;
    }

    LOG_ERROR("Could not find image input description named 'image' or it's not an image type");
    return NO;
}

@end
