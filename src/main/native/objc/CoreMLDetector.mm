#import "CoreMLDetector.h"
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <Vision/Vision.h>

// Objective-C 实现类
@interface CoreMLDetectorImpl : NSObject {
    MLModel* _model;
    MLModelConfiguration* _config;
    float _scaleX;  // 宽度缩放比例
    float _scaleY;  // 高度缩放比例
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
    CVPixelBufferRef imageBuffer;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorMalloc, matimg.cols, matimg.rows, kCVPixelFormatType_32BGRA, (CFDictionaryRef) CFBridgingRetain(options), &imageBuffer) ;
    
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    void *base = CVPixelBufferGetBaseAddress(imageBuffer);
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
    // 记录原始图像尺寸
    std::printf("原始图像尺寸: (%d, %d)\n", image.cols, image.rows);
    
    // 如果是灰度图，转换为RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }
    
    // 创建缩放后的图像用于模型输入
    cv::Mat image_scaled = cv::Mat::zeros(640, 640, CV_8UC3);
    float scaled_height = 640.0f / (image.cols / (float)image.rows);
    _barHeight = (640 - scaled_height) / 2;
    _scaledHeight = scaled_height;
    
    std::printf("缩放后图像尺寸: (%d, %d)\n", image_scaled.cols, image_scaled.rows);
    std::printf("缩放高度: %.2f, 上下填充高度: %d\n", scaled_height, _barHeight);
    
    // 计算缩放后的图像尺寸
    cv::Size scaled_size(640, scaled_height);
    cv::Mat resized;
    cv::resize(image, resized, scaled_size);
    
    // 将缩放后的图像放入中间位置
    resized.copyTo(image_scaled(cv::Rect(0, _barHeight, 640, scaled_height)));
    
    // 保存缩放比例用于后处理
    _scaleX = 1.0f;
    _scaleY = scaled_height / 640.0f;
    
    return image_scaled;
}

- (DetectionResult)processDetectionResult:(float*)coords 
                               confidence:(float*)confs 
                              imageWidth:(int)imageWidth 
                             imageHeight:(int)imageHeight
                              numClasses:(NSInteger)numClasses {
    // 找到最大置信度的类别
    float maxConf = confs[0];
    NSInteger objClass = 0;
    
    for (NSInteger i = 1; i < numClasses; i++) {
        if (confs[i] > maxConf) {
            maxConf = confs[i];
            objClass = i;
        }
    }
    
    std::printf("检测到目标 - 类别: %ld, 置信度: %.2f\n", (long)objClass, maxConf);
    std::printf("原始坐标: x=%.2f, y=%.2f, w=%.2f, h=%.2f\n", 
                coords[0], coords[1], coords[2], coords[3]);
    
    // 获取归一化的坐标值
    float x = coords[0] * imageWidth;
    float y = ((coords[1] * 640 - _barHeight) / _scaledHeight) * imageHeight;
    float width = coords[2] * imageWidth;
    float height = coords[3] / (_scaledHeight / 640.0f) * imageHeight;
    
    std::printf("映射后坐标: x=%.2f, y=%.2f, w=%.2f, h=%.2f\n", x, y, width, height);
    
    // 计算边界框的四个角点
    float x1 = x - width/2;
    float y1 = y - height/2;
    float x2 = x + width/2;
    float y2 = y + height/2;
    
    // 创建检测结果
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
    // 保存原始图像尺寸
    int originalWidth = image.cols;
    int originalHeight = image.rows;
    
    // 前处理：缩放图像到640x640
    cv::Mat resizedImage = [self preprocessImage:image];
    
    // 将 OpenCV Mat 转换为 CVPixelBuffer
    CVPixelBufferRef pixelBuffer = getImageBufferFromMat(resizedImage);
    
    // 创建 MLFeatureValue
    NSError* error = nil;
    MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
    
    CVPixelBufferRelease(pixelBuffer);
    
    // 创建输入特征
    NSDictionary* inputFeatures = @{@"image": imageFeatureValue};
    MLDictionaryFeatureProvider* input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputFeatures error:&error];
    
    if (error) {
        NSLog(@"Error creating input features: %@", error);
        return nil;
    }
    
    // 进行预测
    id<MLFeatureProvider> output = [_model predictionFromFeatures:input error:&error];
    
    if (error) {
        NSLog(@"Prediction error: %@", error);
        return nil;
    }

    // 获取坐标和置信度
    MLFeatureValue* coordinatesValue = [output featureValueForName:@"coordinates"];
    MLFeatureValue* confidenceValue = [output featureValueForName:@"confidence"];
    
    if (!coordinatesValue || !confidenceValue) {
        std::printf("Failed to get coordinates or confidence from output");
        return nil;
    }
    
    MLMultiArray* coordinates = coordinatesValue.multiArrayValue;
    MLMultiArray* confidence = confidenceValue.multiArrayValue;
    
    if (!coordinates || !confidence) {
        std::printf("Failed to get multiarray from feature values");
        return nil;
    }
    
    // 获取类别数量
    NSInteger numClasses = [confidence.shape[1] integerValue];
    
    // 创建检测结果数组
    NSMutableArray* results = [NSMutableArray array];
    
    // 获取坐标和置信度值
    float* coords = (float*)coordinates.dataPointer;
    float* confs = (float*)confidence.dataPointer;
    
    // 处理后处理
    DetectionResult result = [self processDetectionResult:coords 
                                              confidence:confs 
                                             imageWidth:originalWidth 
                                            imageHeight:originalHeight
                                             numClasses:numClasses];
    
    // 将结果包装为NSValue并添加到数组
    NSValue* value = [NSValue valueWithBytes:&result objCType:@encode(DetectionResult)];
    [results addObject:value];
    
    return results;
}

@end

// C++ 实现
CoreMLDetector::CoreMLDetector(const std::string& modelPath) {
    NSString* nsModelPath = [NSString stringWithUTF8String:modelPath.c_str()];
    impl_ = ( void*)[[CoreMLDetectorImpl alloc] initWithModelPath:nsModelPath];
}

CoreMLDetector::~CoreMLDetector() {
    if (impl_) {
        CoreMLDetectorImpl* obj = (CoreMLDetectorImpl*)impl_;
        obj = nil;
    }
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