#import "CoreMLDetector.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

#define LOG_INFO(fmt, ...) NSLog((@"[INFO][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) NSLog((@"[ERROR][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) NSLog((@"[DEBUG][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)
#define LOG_PERF(fmt, ...) NSLog((@"[PERF][%s:%d] " fmt), __FUNCTION__, __LINE__, ##__VA_ARGS__)    

@interface CoreMLDetectorImpl : NSObject {
    MLModel* _model;
    MLModelConfiguration* _config;
    float _scaleX;  // width scale
    float _scaleY;  // height scale
    int _barHeight;
    float _scaledHeight;
}

- (instancetype)initWithModelPath:(NSString *)modelPath;
- (NSArray *)detect:(cv::Mat)image nmsThresh:(double)nmsThresh boxThresh:(double)boxThresh;
- (void)setCoreMask:(int)coreMask;
- (DetectionResult)processDetectionResult:(float*)coords confidence:(float*)confs imageWidth:(int)imageWidth imageHeight:(int)imageHeight numClasses:(NSInteger)numClasses;
- (cv::Mat)preprocessImage:(cv::Mat)image;

@end


CVPixelBufferRef getImageBufferFromMat(cv::Mat matimg) {
    cv::cvtColor(matimg, matimg, cv::COLOR_BGR2BGRA);
    
    int widthReminder = matimg.cols % 64, heightReminder = matimg.rows % 64;
    if (widthReminder != 0 || heightReminder != 0) {
        cv::resize(matimg, matimg, cv::Size(matimg.cols + (64 - widthReminder), matimg.rows + (64 - heightReminder)));
    }

    NSDictionary *options = [NSDictionary dictionaryWithObjectsAndKeys:
                                [NSNumber numberWithBool: YES], kCVPixelBufferMetalCompatibilityKey,
                                [NSNumber numberWithBool: YES], kCVPixelBufferCGImageCompatibilityKey,
                                [NSNumber numberWithBool: YES], kCVPixelBufferCGBitmapContextCompatibilityKey,
                                [NSNumber numberWithInt: matimg.cols], kCVPixelBufferWidthKey,
                                [NSNumber numberWithInt: matimg.rows], kCVPixelBufferHeightKey,
                                [NSNumber numberWithInt: matimg.step[0]], kCVPixelBufferBytesPerRowAlignmentKey,
                                nil];
    CVPixelBufferRef imageBuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorMalloc, matimg.cols, matimg.rows, kCVPixelFormatType_32BGRA, (CFDictionaryRef) CFBridgingRetain(options), &imageBuffer) ;
    if (status != kCVReturnSuccess || imageBuffer == NULL) {
        LOG_ERROR("Failed to create CVPixelBuffer in getImageBufferFromMat");
        if (imageBuffer) CVPixelBufferRelease(imageBuffer);
        return NULL;
    }
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    void *base = CVPixelBufferGetBaseAddress(imageBuffer);
    if (!base) {
        LOG_ERROR("Failed to get base address for CVPixelBuffer");
        CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
        CVPixelBufferRelease(imageBuffer);
        return NULL;
    }
    memcpy(base, matimg.data, matimg.total() * matimg.elemSize());
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);

    return imageBuffer;
} 

@implementation CoreMLDetectorImpl

- (instancetype)initWithModelPath:(NSString *)modelPath {
    self = [super init];
    if (self) {
        NSURL* modelURL = [NSURL fileURLWithPath:modelPath];
        NSURL* compiledURL = [MLModel compileModelAtURL:modelURL error:nil];
        
        _config = [[MLModelConfiguration alloc] init];
        _model = [MLModel modelWithContentsOfURL:compiledURL configuration:_config error:nil];
    }
    return self;
}

- (void)setCoreMask:(int)coreMask {
    _config.computeUnits = MLComputeUnitsAll;
}

- (cv::Mat)preprocessImage:(cv::Mat)image {
    // Check if the input image is valid
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        LOG_ERROR("Invalid input image");
        return cv::Mat();
    }
    
    // If it is a grayscale image, convert it to RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }
    
    // Create a scaled image for model input
    cv::Mat image_scaled = cv::Mat::zeros(640, 640, CV_8UC3);
    
    // Calculate the scaled height to avoid extreme values
    float aspect_ratio = (float)image.rows / (float)image.cols;
    float scaled_height = 640.0f * aspect_ratio;
    
    // Limit the scaled height to a reasonable range
    scaled_height = std::min(std::max(scaled_height, 1.0f), 640.0f);
    _barHeight = (640 - scaled_height) / 2;
    _scaledHeight = scaled_height;
    
    // Calculate the size of the scaled image
    cv::Size scaled_size(640, scaled_height);
    cv::Mat resized;
    cv::resize(image, resized, scaled_size);
    
    // Ensure the target area is within a valid range
    int valid_height = std::min((int)scaled_height, 640 - _barHeight);
    if (_barHeight >= 0 && _barHeight < 640 && valid_height > 0) {
        resized.copyTo(image_scaled(cv::Rect(0, _barHeight, 640, valid_height)));
    } else {
        LOG_ERROR("Invalid padding or scaled height calculated");
        return cv::Mat();
    }
    
    // Save the scaling ratio for post-processing
    _scaleX = 1.0f;
    _scaleY = scaled_height / 640.0f;
    
    return image_scaled;
}

- (DetectionResult)processDetectionResult:(float*)coords 
                               confidence:(float*)confs 
                              imageWidth:(int)imageWidth 
                             imageHeight:(int)imageHeight
                              numClasses:(NSInteger)numClasses {
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

    // Get the normalized coordinate values
    float x = coords[0] * imageWidth;
    float y = ((coords[1] * 640 - _barHeight) / _scaledHeight) * imageHeight;
    float width = coords[2] * imageWidth;
    float height = coords[3] / (_scaledHeight / 640.0f) * imageHeight;
    
    // Verify that the coordinate values are within a valid range
    if (x < 0 || y < 0 || width <= 0 || height <= 0) {
        LOG_ERROR("Invalid coordinate values");
        DetectionResult emptyResult;
        memset(&emptyResult, 0, sizeof(DetectionResult));
        return emptyResult;
    }
    
    // Calculate the four corners of the bounding box
    float x1 = x - width/2;
    float y1 = y - height/2;
    float x2 = x + width/2;
    float y2 = y + height/2;
    
    // Ensure the coordinates are within the image range
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
    
    // Preprocessing: scale the image to 640x640
    cv::Mat resizedImage = [self preprocessImage:image];
    if (resizedImage.empty()) {
        LOG_ERROR("Image preprocessing failed");
        return [NSArray array];
    }
    
    NSTimeInterval preprocessTime = -[startTime timeIntervalSinceNow];
    LOG_PERF("Preprocess time: %.3f ms", preprocessTime * 1000);
    
    // Convert OpenCV Mat to CVPixelBuffer
    CVPixelBufferRef pixelBuffer = getImageBufferFromMat(resizedImage);
    if (!pixelBuffer) {
        LOG_ERROR("Failed to create CVPixelBuffer");
        return [NSArray array];
    }
    
    // Create MLFeatureValue
    NSError* error = nil;
    MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
    if (!imageFeatureValue) {
        LOG_ERROR("Failed to create MLFeatureValue");
        CVPixelBufferRelease(pixelBuffer);
        return [NSArray array];
    }
    
    // Create input features
    NSDictionary* inputFeatures = @{@"image": imageFeatureValue};
    MLDictionaryFeatureProvider* input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputFeatures error:&error];
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
        LOG_ERROR("Invalid number of boxes or classes");
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
                                                 imageWidth:originalWidth
                                                imageHeight:originalHeight
                                                 numClasses:numClasses];
        if (result.confidence > boxThresh) {
            NSValue* value = [NSValue valueWithBytes:&result objCType:@encode(DetectionResult)];
            [results addObject:value];
        }
    }
    
    // 记录后处理耗时
    NSTimeInterval postprocessTime = -[postprocessStartTime timeIntervalSinceNow];
    LOG_PERF("Postprocess time: %.3f ms", postprocessTime * 1000);
    
    // 记录总耗时
    NSTimeInterval totalTime = -[startTime timeIntervalSinceNow];
    LOG_PERF("Total processing time: %.3f ms", totalTime * 1000);

    return results;
}

@end

// C++ Implementation
CoreMLDetector::CoreMLDetector(const std::string& modelPath) {
    NSString* nsModelPath = [NSString stringWithUTF8String:modelPath.c_str()];
    impl_ = ( void*)[[CoreMLDetectorImpl alloc] initWithModelPath:nsModelPath];
}

CoreMLDetector::~CoreMLDetector() {
    // ARC will manage Objective-C object memory, no need to set to nil
}

void CoreMLDetector::setCoreMask(int coreMask) {
    CoreMLDetectorImpl* obj = (__bridge CoreMLDetectorImpl*)impl_;
    [obj setCoreMask:coreMask];
}

std::vector<DetectionResult> CoreMLDetector::detect(const cv::Mat& image, double nmsThresh, double boxThresh) {
    CoreMLDetectorImpl* obj = (__bridge CoreMLDetectorImpl*)impl_;
    NSArray* results = [obj detect:image nmsThresh:nmsThresh boxThresh:boxThresh];
    
    std::vector<DetectionResult> detections;
    if (results) {
        for (NSValue* resultValue in results) {
            DetectionResult result;
            [resultValue getValue:&result];
            detections.push_back(result);
        }
    }
    
    return detections;
} 