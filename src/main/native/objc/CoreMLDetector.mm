#import "CoreMLDetector.h"
#import "PixelBufferPool.h"
#import "CoreMLUtils.h"
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

@interface CoreMLDetectorImpl : NSObject {
    MLModel* _model;
    MLModelConfiguration* _config;
    NSInteger _inputWidth;
    NSInteger _inputHeight;
}

- (instancetype)initWithModelPath:(NSString *)modelPath;
- (NSArray *)detect:(cv::Mat)image nmsThresh:(double)nmsThresh boxThresh:(double)boxThresh;
- (int)setCoreMask:(int)coreMask;

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
            _inputWidth = imageConstraint.pixelsWide;
            _inputHeight = imageConstraint.pixelsHigh;

            if (_inputWidth <= 0 || _inputHeight <= 0) {
                 LOG_ERROR("Invalid input dimensions retrieved from model: %ld x %ld", _inputWidth, _inputHeight);
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



- (NSArray *)detect:(cv::Mat)image nmsThresh:(double)nmsThresh boxThresh:(double)boxThresh {
    @autoreleasepool {
        NSDate *startTime = [NSDate date];

        // Save the original image size
        int originalWidth = image.cols;
        int originalHeight = image.rows;

        // Get model input dimensions
        PreprocessParams params = {0};
        if (![CoreMLUtils getModelInputDimensions:_model params:&params]) {
            LOG_ERROR("Failed to get model input dimensions");
            return @[];
        }

        // Preprocessing: scale the image to input dimensions
        cv::Mat resizedImage = [CoreMLUtils preprocessImage:image params:&params];
        if (resizedImage.empty()) {
            LOG_ERROR("Image preprocessing failed");
            return @[];
        }

        NSTimeInterval preprocessTime = -[startTime timeIntervalSinceNow];
        LOG_PERF("Preprocess time: %.3f ms", preprocessTime * 1000);

        // Convert OpenCV Mat to CVPixelBuffer using the pixel buffer pool
        CVPixelBufferRef pixelBuffer = [CoreMLUtils matToCVPixelBuffer:resizedImage];
        if (!pixelBuffer) {
            LOG_ERROR("Failed to create CVPixelBuffer");
            return @[];
        }

        // Get model input description to verify expected inputs
        MLModelDescription *modelDescription = _model.modelDescription;
        NSDictionary<NSString *, MLFeatureDescription *> *inputsDesc = modelDescription.inputDescriptionsByName;

        // Create MLFeatureValue
        NSError* error = nil;
        MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];

        // Build input feature dictionary dynamically
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
            // Return the pixel buffer to the pool instead of releasing it
            [[PixelBufferPool sharedPool] returnPixelBuffer:pixelBuffer];
            return @[];
        }

        NSDate *inferenceStartTime = [NSDate date];

        // Run prediction
        id<MLFeatureProvider> output = [_model predictionFromFeatures:input error:&error];
        // Return the pixel buffer to the pool instead of releasing it
        [[PixelBufferPool sharedPool] returnPixelBuffer:pixelBuffer];

        if (error) {
            LOG_ERROR("Prediction error: %@", error);
            return @[];
        }

        NSTimeInterval inferenceTime = -[inferenceStartTime timeIntervalSinceNow];
        LOG_PERF("Model inference time: %.3f ms", inferenceTime * 1000);

        // Get coordinates and confidence
        MLFeatureValue* coordinatesValue = [output featureValueForName:@"coordinates"];
        MLFeatureValue* confidenceValue = [output featureValueForName:@"confidence"];

        if (!coordinatesValue || !confidenceValue) {
            LOG_ERROR("Failed to get coordinates or confidence output");
            return @[];
        }

        MLMultiArray* coordinates = coordinatesValue.multiArrayValue;
        MLMultiArray* confidence = confidenceValue.multiArrayValue;

        if (!coordinates || !confidence) {
            LOG_ERROR("Failed to get multi-array values");
            return @[];
        }

        NSInteger numBoxes = [coordinates.shape[0] integerValue];   // box num
        NSInteger numClasses = [confidence.shape[1] integerValue];  // class num
        if (numBoxes <= 0 || numClasses <= 0) {
            return @[];
        }

        NSDate *postprocessStartTime = [NSDate date];

        // Create the detection results array
        NSMutableArray* results = [NSMutableArray array];

        // Get coordinate and confidence values
        float* coords = (float*)coordinates.dataPointer;
        float* confs = (float*)confidence.dataPointer;

        if (!coords || !confs) {
            LOG_ERROR("Failed to get coordinates or confidence data");
            return @[];
        }

        for (NSInteger i = 0; i < numBoxes; i++) {
            float* boxCoords = coords + i * 4; // Each box has 4 coordinates
            float* boxConfs = confs + i * numClasses; // Each box has numClasses confidences

            // Use CoreMLUtils to process detection results
            DetectionResult result = [CoreMLUtils processDetectionResult:boxCoords
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

        NSArray* retainedResults = [results copy];
        return retainedResults;
    }
}

@end

// C++ Implementation
CoreMLDetector::CoreMLDetector(const std::string& modelPath) {
    NSString* nsModelPath = [NSString stringWithUTF8String:modelPath.c_str()];
    CoreMLDetectorImpl* detector = [[CoreMLDetectorImpl alloc] initWithModelPath:nsModelPath];
    if (detector == nil) {
        throw std::runtime_error("Failed to initialize CoreMLDetector");
    }
    impl_ = (__bridge_retained void*)detector;
}

CoreMLDetector::~CoreMLDetector() {
    if (impl_) {
        CoreMLDetectorImpl* obj = (__bridge_transfer CoreMLDetectorImpl*)impl_;
        impl_ = nullptr;
    }
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