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
    NSInteger _inputWidth; // Model input width
    NSInteger _inputHeight; // Model input height
    float _scaleFactor; // Scale factor used in preprocessing
    float _padWidth;    // Width padding used in preprocessing
    float _padHeight;   // Height padding used in preprocessing
}

- (instancetype)initWithModelPath:(NSString *)modelPath;
- (NSArray *)detect:(cv::Mat)image nmsThresh:(double)nmsThresh boxThresh:(double)boxThresh;
- (int)setCoreMask:(int)coreMask;
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
    CVPixelBufferRef imageBuffer = nullptr;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorMalloc, matimg.cols, matimg.rows, kCVPixelFormatType_32BGRA, (CFDictionaryRef) CFBridgingRetain(options), &imageBuffer) ;
    if (status != kCVReturnSuccess || imageBuffer == nullptr) {
        LOG_ERROR("Failed to create CVPixelBuffer in getImageBufferFromMat");
        if (imageBuffer) {
            CVPixelBufferRelease(imageBuffer);
        }
        return nullptr;
    }
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    void *base = CVPixelBufferGetBaseAddress(imageBuffer);
    if (!base) {
        LOG_ERROR("Failed to get base address for CVPixelBuffer");
        CVPixelBufferUnlockBaseAddress(imageBuffer, 0);
        CVPixelBufferRelease(imageBuffer);
        return nullptr;
    }
    memcpy(base, matimg.data, matimg.total() * matimg.elemSize());
    CVPixelBufferUnlockBaseAddress(imageBuffer, 0);

    return imageBuffer;
} 

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
        // Assuming the image input name is "image". Adjust if necessary.
        MLFeatureDescription *imageInputDescription = inputDescriptions[@"image"]; 
        
        if (imageInputDescription && imageInputDescription.type == MLFeatureTypeImage) {
            MLImageConstraint *imageConstraint = imageInputDescription.imageConstraint;
            _inputWidth = imageConstraint.pixelsWide;
            _inputHeight = imageConstraint.pixelsHigh;

            if (_inputWidth <= 0 || _inputHeight <= 0) {
                 LOG_ERROR("Invalid input dimensions retrieved from model: %ld x %ld", _inputWidth, _inputHeight);
                 // Fallback to default or handle error appropriately
                 // For now, let's return nil if we can't get valid dimensions
                 return nil;
            }
        } else {
            LOG_ERROR("Could not find image input description named 'image' or it's not an image type.");
            // Handle error: maybe fallback to a default size or return nil
            // For now, let's return nil
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
    
    // --- Preprocessing with dynamic input size ---
    // Calculate scaling factor to fit within _inputWidth x _inputHeight while preserving aspect ratio
    _scaleFactor = std::min((float)_inputWidth / image.cols, (float)_inputHeight / image.rows);
    
    // Calculate new dimensions after scaling
    int new_w = round(image.cols * _scaleFactor);
    int new_h = round(image.rows * _scaleFactor);
    
    // Ensure new dimensions are valid
    if (new_w <= 0 || new_h <= 0) {
        LOG_ERROR("Invalid scaled dimensions calculated: %d x %d", new_w, new_h);
        return cv::Mat();
    }

    // Calculate padding
    _padWidth = (_inputWidth - new_w) / 2.0f;
    _padHeight = (_inputHeight - new_h) / 2.0f;

    // Resize the image
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_w, new_h));

    // Create the final scaled/padded image buffer for the model
    cv::Mat image_scaled = cv::Mat::zeros(_inputHeight, _inputWidth, CV_8UC3);

    // Define the Region of Interest (ROI) in the scaled image where the resized image will be copied
    cv::Rect roi(round(_padWidth), round(_padHeight), new_w, new_h);

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

    // --- Coordinate Transformation based on new preprocessing ---
    // Model outputs coordinates relative to the _inputWidth x _inputHeight padded image
    // coords[0] = center_x, coords[1] = center_y, coords[2] = width, coords[3] = height (normalized 0-1)
    
    // Convert normalized coordinates to absolute coordinates in the scaled/padded image space
    float abs_cx = coords[0] * _inputWidth;
    float abs_cy = coords[1] * _inputHeight;
    float abs_w = coords[2] * _inputWidth;
    float abs_h = coords[3] * _inputHeight;

    // Revert padding and scaling to get coordinates in the original image space
    float orig_cx = (abs_cx - _padWidth) / _scaleFactor;
    float orig_cy = (abs_cy - _padHeight) / _scaleFactor;
    float orig_w = abs_w / _scaleFactor;
    float orig_h = abs_h / _scaleFactor;

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
    
    // --- Adjust MLFeatureValue creation if necessary ---
    // Check if the model actually expects explicit iou/confidence thresholds as input
    // Many detection models (like YOLO) handle this internally or in post-processing layers.
    // If they are NOT part of the model's expected input (check modelDescription.inputDescriptionsByName), remove them.
    
    // Get model input description again to verify expected inputs
    MLModelDescription *modelDesc = _model.modelDescription;
    NSDictionary<NSString *, MLFeatureDescription *> *inputsDesc = modelDesc.inputDescriptionsByName;

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
                                                numClasses:numClasses];
        
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
    impl_ = (void*)detector;
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