#import "CoreMLDetector.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

// Define Log Levels
#define LOG_LEVEL_NONE  0
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_INFO  2
#define LOG_LEVEL_PERF  3
#define LOG_LEVEL_DEBUG 4

// Set default log level if not defined by compiler/CMake
#ifndef CURRENT_LOG_LEVEL
#define CURRENT_LOG_LEVEL LOG_LEVEL_INFO // Default to INFO level
#endif

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

#if CURRENT_LOG_LEVEL >= LOG_LEVEL_PERF
#define LOG_PERF(fmt, ...) NSLog((@"[PERF][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_PERF(fmt, ...) do {} while(0)
#endif

#if CURRENT_LOG_LEVEL >= LOG_LEVEL_DEBUG
#define LOG_DEBUG(fmt, ...) NSLog((@"[DEBUG][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOG_DEBUG(fmt, ...) do {} while(0)
#endif

// Add PreprocessParams struct before the CoreMLDetectorImpl interface
using PreprocessParams = struct {
    NSInteger inputWidth;
    NSInteger inputHeight;
    float scaleFactor;
    float padWidth;
    float padHeight;
};

@interface CoreMLDetectorImpl : NSObject {
    MLModel* _model;
    MLModelConfiguration* _config;
    CVPixelBufferPoolRef _pixelBufferPool;
    NSInteger _poolWidth;
    NSInteger _poolHeight;
}

- (instancetype)initWithModelPath:(NSString *)modelPath;
- (NSArray *)detect:(cv::Mat)image nmsThresh:(double)nmsThresh boxThresh:(double)boxThresh;
- (int)setCoreMask:(int)coreMask;
- (DetectionResult)processDetectionResult:(float*)coords 
                               confidence:(float*)confs 
                              imageWidth:(int)imageWidth 
                             imageHeight:(int)imageHeight
                              numClasses:(NSInteger)numClasses
                              params:(PreprocessParams)params;
- (cv::Mat)preprocessImage:(cv::Mat)image params:(PreprocessParams*)params;
- (CVPixelBufferRef)getImageBufferFromMat:(cv::Mat)matimg;

@end

@implementation CoreMLDetectorImpl

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    NSError* error = nil;

    if (self) {
        NSURL* modelURL = [NSURL fileURLWithPath:modelPath];
        NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:&error];
        
        _config = [[MLModelConfiguration alloc] init];
        _model = [MLModel modelWithContentsOfURL:compiledURL configuration:_config error:&error];

        if (!_model) {
            LOG_ERROR("Error creating MLModel: %@", error);
            return nil;
        }

        // Get model description and input image size
        MLModelDescription *modelDescription = _model.modelDescription;
        NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptions = modelDescription.inputDescriptionsByName;
        MLFeatureDescription *imageInputDescription = inputDescriptions[@"image"]; 
        
        if (imageInputDescription && imageInputDescription.type == MLFeatureTypeImage) {
            MLImageConstraint *imageConstraint = imageInputDescription.imageConstraint;
            _poolWidth = imageConstraint.pixelsWide;
            _poolHeight = imageConstraint.pixelsHigh;

            if (_poolWidth <= 0 || _poolHeight <= 0) {
                 LOG_ERROR("Invalid input dimensions retrieved from model: %ld x %ld", _poolWidth, _poolHeight);
                 return nil;
            }

            // Create pixel buffer pool
            NSDictionary *poolAttributes = @{
                (NSString *)kCVPixelBufferPoolMinimumBufferCountKey: @10,
                (NSString *)kCVPixelBufferPoolMaximumBufferAgeKey: @(10.0)
            };

            NSDictionary *pixelBufferAttributes = @{
                (NSString *)kCVPixelBufferMetalCompatibilityKey: @YES,
                (NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
                (NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
                (NSString *)kCVPixelBufferWidthKey: @(_poolWidth),
                (NSString *)kCVPixelBufferHeightKey: @(_poolHeight),
                (NSString *)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA),
                (NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{}
            };

            CVReturn status = CVPixelBufferPoolCreate(kCFAllocatorDefault,
                                                    (__bridge CFDictionaryRef)poolAttributes,
                                                    (__bridge CFDictionaryRef)pixelBufferAttributes,
                                                    &_pixelBufferPool);

            if (status != kCVReturnSuccess || !_pixelBufferPool) {
                LOG_ERROR("Failed to create pixel buffer pool: %d", status);
                return nil;
            }
        } else {
            LOG_ERROR("Could not find image input description named 'image' or it's not an image type.");
            return nil;
        }
    }

    if (error) {
        LOG_ERROR("Error creating MLModel: %@", error);
        return nil;
    }

    return self;
}

- (void)dealloc {
    if (_pixelBufferPool) {
        CVPixelBufferPoolRelease(_pixelBufferPool);
        _pixelBufferPool = nullptr;
    }
    [super dealloc];
}

- (CVPixelBufferRef)getImageBufferFromMat:(cv::Mat)matimg {
    cv::cvtColor(matimg, matimg, cv::COLOR_BGR2BGRA);
    
    int widthReminder = matimg.cols % 64, heightReminder = matimg.rows % 64;
    if (widthReminder != 0 || heightReminder != 0) {
        cv::resize(matimg, matimg, cv::Size(matimg.cols + (64 - widthReminder), matimg.rows + (64 - heightReminder)));
    }

    // Try to get a buffer from the pool
    CVPixelBufferRef imageBuffer = nullptr;
    CVReturn status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, _pixelBufferPool, &imageBuffer);
    
    if (status != kCVReturnSuccess || !imageBuffer) {
        LOG_ERROR("Failed to get pixel buffer from pool: %d", status);
        return nullptr;
    }

    // Lock the buffer for writing
    status = CVPixelBufferLockBaseAddress(imageBuffer, 0);
    if (status != kCVReturnSuccess) {
        LOG_ERROR("Failed to lock pixel buffer: %d", status);
        CVPixelBufferRelease(imageBuffer);
        return nullptr;
    }

    void *base = CVPixelBufferGetBaseAddress(imageBuffer);
    if (!base) {
        LOG_ERROR("Failed to get base address for pixel buffer");
        CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
        CVPixelBufferRelease(imageBuffer);
        return nullptr;
    }

    // Copy the image data
    memcpy(base, matimg.data, matimg.total() * matimg.elemSize());
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);

    return imageBuffer;
}

// Set the compute units for the model, return YES if successful, NO otherwise
- (int)setCoreMask:(int)coreMask {
    @try {
        _config.computeUnits = (MLComputeUnits)coreMask;
        return 0; // Success
    } @catch (NSException *exception) {
        LOG_ERROR("Exception in setCoreMask: %s", [[exception reason] UTF8String]);
        return -1; // Failure
    }
}

- (cv::Mat)preprocessImage:(cv::Mat)image params:(PreprocessParams*)params {
    // Check if the input image is valid
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        LOG_ERROR("Invalid input image");
        return cv::Mat();
    }
    
    // If it is a grayscale image, convert it to RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }
    
    // --- Preprocessing with dynamic input size ---
    // Calculate scaling factor to fit within inputWidth x inputHeight while preserving aspect ratio
    params->scaleFactor = std::min((float)params->inputWidth / image.cols, (float)params->inputHeight / image.rows);
    
    // Calculate new dimensions after scaling
    int new_w = round(image.cols * params->scaleFactor);
    int new_h = round(image.rows * params->scaleFactor);
    
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
    cv::resize(image, resized, cv::Size(new_w, new_h));

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

- (DetectionResult)processDetectionResult:(float*)coords 
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

    // --- Coordinate Transformation based on new preprocessing ---
    // Model outputs coordinates relative to the _inputWidth x _inputHeight padded image
    // coords[0] = center_x, coords[1] = center_y, coords[2] = width, coords[3] = height (normalized 0-1)
    
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

- (NSArray *)detect:(cv::Mat)image nmsThresh:(double)nmsThresh boxThresh:(double)boxThresh {
    NSDate *startTime = [NSDate date];
    
    // Save the original image size
    int originalWidth = image.cols;
    int originalHeight = image.rows;
    
    // Get model input dimensions
    MLModelDescription *modelDescription = _model.modelDescription;
    NSDictionary<NSString *, MLFeatureDescription *> *inputDescriptions = modelDescription.inputDescriptionsByName;
    MLFeatureDescription *imageInputDescription = inputDescriptions[@"image"];
    
    PreprocessParams params = {0};
    
    if (imageInputDescription && imageInputDescription.type == MLFeatureTypeImage) {
        MLImageConstraint *imageConstraint = imageInputDescription.imageConstraint;
        params.inputWidth = imageConstraint.pixelsWide;
        params.inputHeight = imageConstraint.pixelsHigh;
    }
    
    if (params.inputWidth <= 0 || params.inputHeight <= 0) {
        LOG_ERROR("Invalid input dimensions retrieved from model: %ld x %ld", params.inputWidth, params.inputHeight);
        return [NSArray array];
    }
    
    // Preprocessing: scale the image to input dimensions
    cv::Mat resizedImage = [self preprocessImage:image params:&params];
    if (resizedImage.empty()) {
        LOG_ERROR("Image preprocessing failed");
        return [NSArray array];
    }
    
    NSTimeInterval preprocessTime = -[startTime timeIntervalSinceNow];
    LOG_PERF("Preprocess time: %.3f ms", preprocessTime * 1000);
    
    // Convert OpenCV Mat to CVPixelBuffer
    CVPixelBufferRef pixelBuffer = [self getImageBufferFromMat:resizedImage];
    if (!pixelBuffer) {
        LOG_ERROR("Failed to create CVPixelBuffer");
        return [NSArray array];
    }
    
    // Get model input description again to verify expected inputs
    NSDictionary<NSString *, MLFeatureDescription *> *inputsDesc = modelDescription.inputDescriptionsByName;
    
    // Create MLFeatureValue
    NSError* error = nil;
    MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
    
    // Build input feature dictionary Dynamically
    NSMutableDictionary* inputFeatures = [NSMutableDictionary dictionary];
    inputFeatures[@"image"] = imageFeatureValue; // Assuming "image" is the correct input name

    // Conditionally add other inputs if the model expects them
    if (inputsDesc[@"iouThreshold"]) {
        MLFeatureValue* iouThresholdValue = [MLFeatureValue featureValueWithDouble:nmsThresh];
        inputFeatures[@"iouThreshold"] = iouThresholdValue;
    }
    
    if (inputsDesc[@"confidenceThreshold"]) {
        MLFeatureValue* confidenceThresholdValue = [MLFeatureValue featureValueWithDouble:boxThresh];
        inputFeatures[@"confidenceThreshold"] = confidenceThresholdValue;
    }
    
    MLDictionaryFeatureProvider* input = [[MLDictionaryFeatureProvider alloc] 
                                        initWithDictionary:inputFeatures error:&error];
    
    if (error) {
        LOG_ERROR("Error creating input features: %@", error);
        CVPixelBufferRelease(pixelBuffer);
        return [NSArray array];
    }
    
    NSDate *inferenceStartTime = [NSDate date];
    
    // Run prediction
    id<MLFeatureProvider> output = [_model predictionFromFeatures:input error:&error];
    CVPixelBufferRelease(pixelBuffer);
    if (error) {
        LOG_ERROR("Prediction error: %@", error);
        return [NSArray array];
    }
    
    NSTimeInterval inferenceTime = -[inferenceStartTime timeIntervalSinceNow];
    LOG_PERF("Model inference time: %.3f ms", inferenceTime * 1000);

    // Get coordinates and confidence
    MLFeatureValue* coordinatesValue = [output featureValueForName:@"coordinates"];
    MLFeatureValue* confidenceValue = [output featureValueForName:@"confidence"];
    
    if (!coordinatesValue || !confidenceValue) {
        LOG_ERROR("Failed to get coordinates or confidence output");
        return [NSArray array];
    }
    
    MLMultiArray* coordinates = coordinatesValue.multiArrayValue;
    MLMultiArray* confidence = confidenceValue.multiArrayValue;
    
    if (!coordinates || !confidence) {
        LOG_ERROR("Failed to get multi-array values");
        return [NSArray array];
    }

    NSInteger numBoxes = [coordinates.shape[0] integerValue];   // box num
    NSInteger numClasses = [confidence.shape[1] integerValue];  // class num
    if (numBoxes <= 0 || numClasses <= 0) {
        return [NSArray array];
    }
    
    NSDate *postprocessStartTime = [NSDate date];
    
    // Create the detection results array
    NSMutableArray* results = [NSMutableArray array];
    
    // Get coordinate and confidence values
    float* coords = (float*)coordinates.dataPointer;
    float* confs = (float*)confidence.dataPointer;
    
    if (!coords || !confs) {
        LOG_ERROR("Failed to get coordinates or confidence data");
        return [NSArray array];
    }
    for (NSInteger i = 0; i < numBoxes; i++) {
        float* boxCoords = coords + i * 4; // Each box has 4 coordinates
        float* boxConfs = confs + i * numClasses; // Each box has numClasses confidences
        DetectionResult result = [self processDetectionResult:boxCoords
                                                 confidence:boxConfs
                                                imageWidth:image.cols
                                               imageHeight:image.rows
                                                numClasses:numClasses
                                                    params:params];
        
        NSValue* value = [NSValue valueWithBytes:&result objCType:@encode(DetectionResult)];
        [results addObject:value];
    }
    
    // Record post-processing time
    NSTimeInterval postprocessTime = -[postprocessStartTime timeIntervalSinceNow];
    LOG_PERF("Postprocess time: %.3f ms", postprocessTime * 1000);
    
    // Record total processing time
    NSTimeInterval totalTime = -[startTime timeIntervalSinceNow];
    LOG_PERF("Total processing time: %.3f ms", totalTime * 1000);

    return results;
}

@end

// C++ Implementation
CoreMLDetector::CoreMLDetector(const std::string& modelPath) {
    NSString* nsModelPath = [NSString stringWithUTF8String:modelPath.c_str()];
    CoreMLDetectorImpl* detector = [[CoreMLDetectorImpl alloc] initWithModelPath:nsModelPath];
    if (detector == nil) {
        throw std::runtime_error("Failed to initialize CoreMLDetector");
    }
    impl_ = (__bridge void*)detector;
}

CoreMLDetector::~CoreMLDetector() {
    impl_ = nullptr;
}

int CoreMLDetector::setCoreMask(int coreMask) {
    CoreMLDetectorImpl* obj = (__bridge CoreMLDetectorImpl*)impl_;
    return [obj setCoreMask:coreMask];
}

std::vector<DetectionResult> CoreMLDetector::detect(const cv::Mat& image, double nmsThresh, double boxThresh) {
    CoreMLDetectorImpl* obj = (__bridge CoreMLDetectorImpl*)impl_;
    NSArray* results = [obj detect:image nmsThresh:nmsThresh boxThresh:boxThresh];
    
    std::vector<DetectionResult> detections{};
    if (results) {
        for (NSValue* resultValue in results) {
            DetectionResult result;
            [resultValue getValue:&result];
            detections.push_back(result);
        }
    }
    
    return detections;
} 