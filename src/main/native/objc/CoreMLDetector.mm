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
    
    // 检查输入图像是否有效
    if (image.empty() || image.cols <= 0 || image.rows <= 0) {
        std::printf("错误：输入图像无效\n");
        return cv::Mat();
    }
    
    // 如果是灰度图，转换为RGB
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2RGB);
    }
    
    // 创建缩放后的图像用于模型输入
    cv::Mat image_scaled = cv::Mat::zeros(640, 640, CV_8UC3);
    
    // 计算缩放后的高度，确保不会出现极端值
    float aspect_ratio = (float)image.rows / (float)image.cols;
    float scaled_height = 640.0f * aspect_ratio;
    
    // 限制缩放后的高度在合理范围内
    scaled_height = std::min(std::max(scaled_height, 1.0f), 640.0f);
    _barHeight = (640 - scaled_height) / 2;
    _scaledHeight = scaled_height;
    
    std::printf("缩放后图像尺寸: (%d, %d)\n", image_scaled.cols, image_scaled.rows);
    std::printf("缩放高度: %.2f, 上下填充高度: %d\n", scaled_height, _barHeight);
    
    // 计算缩放后的图像尺寸
    cv::Size scaled_size(640, scaled_height);
    cv::Mat resized;
    cv::resize(image, resized, scaled_size);
    
    // 确保目标区域在有效范围内
    int valid_height = std::min((int)scaled_height, 640 - _barHeight);
    if (_barHeight >= 0 && _barHeight < 640 && valid_height > 0) {
        resized.copyTo(image_scaled(cv::Rect(0, _barHeight, 640, valid_height)));
    } else {
        std::printf("错误：计算出的填充高度或缩放高度无效\n");
        return cv::Mat();
    }
    
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
    // 检查输入参数是否有效
    if (!coords || !confs || numClasses <= 0) {
        std::printf("错误：无效的输入参数\n");
        DetectionResult emptyResult;
        memset(&emptyResult, 0, sizeof(DetectionResult));
        return emptyResult;
    }
    
    // 找到最大置信度的类别
    float maxConf = confs[0];
    NSInteger objClass = 0;
    
    for (NSInteger i = 1; i < numClasses; i++) {
        if (confs[i] > maxConf) {
            maxConf = confs[i];
            objClass = i;
        }
    }
    
    // 如果最大置信度小于阈值，认为没有检测到目标
    if (maxConf < 0.1f) {  // 可以根据需要调整阈值
        std::printf("未检测到目标，最大置信度: %.2f\n", maxConf);
        DetectionResult emptyResult;
        memset(&emptyResult, 0, sizeof(DetectionResult));
        return emptyResult;
    }

    std::printf("检测到目标 - 类别: %ld, 置信度: %.2f\n", (long)objClass, maxConf);
    std::printf("原始坐标: x=%.2f, y=%.2f, w=%.2f, h=%.2f\n", 
                coords[0], coords[1], coords[2], coords[3]);
    
    // 获取归一化的坐标值
    float x = coords[0] * imageWidth;
    float y = ((coords[1] * 640 - _barHeight) / _scaledHeight) * imageHeight;
    float width = coords[2] * imageWidth;
    float height = coords[3] / (_scaledHeight / 640.0f) * imageHeight;
    
    // 验证坐标值是否在有效范围内
    if (x < 0 || y < 0 || width <= 0 || height <= 0) {
        std::printf("错误：无效的坐标值\n");
        DetectionResult emptyResult;
        memset(&emptyResult, 0, sizeof(DetectionResult));
        return emptyResult;
    }
    
    std::printf("映射后坐标: x=%.2f, y=%.2f, w=%.2f, h=%.2f\n", x, y, width, height);
    
    // 计算边界框的四个角点
    float x1 = x - width/2;
    float y1 = y - height/2;
    float x2 = x + width/2;
    float y2 = y + height/2;
    
    // 确保坐标在图像范围内
    x1 = std::max(0.0f, std::min(x1, (float)imageWidth));
    y1 = std::max(0.0f, std::min(y1, (float)imageHeight));
    x2 = std::max(0.0f, std::min(x2, (float)imageWidth));
    y2 = std::max(0.0f, std::min(y2, (float)imageHeight));
    
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
    if (resizedImage.empty()) {
        std::printf("错误：图像预处理失败\n");
        return [NSArray array];
    }
    
    // 将 OpenCV Mat 转换为 CVPixelBuffer
    CVPixelBufferRef pixelBuffer = getImageBufferFromMat(resizedImage);
    if (!pixelBuffer) {
        std::printf("错误：无法创建 CVPixelBuffer\n");
        return [NSArray array];
    }
    
    // 创建 MLFeatureValue
    NSError* error = nil;
    MLFeatureValue* imageFeatureValue = [MLFeatureValue featureValueWithPixelBuffer:pixelBuffer];
    
    CVPixelBufferRelease(pixelBuffer);
    
    if (!imageFeatureValue) {
        std::printf("错误：无法创建 MLFeatureValue\n");
        return [NSArray array];
    }
    
    // 创建输入特征
    NSDictionary* inputFeatures = @{@"image": imageFeatureValue};
    MLDictionaryFeatureProvider* input = [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputFeatures error:&error];
    
    if (error) {
        NSLog(@"Error creating input features: %@", error);
        return [NSArray array];
    }
    
    // 进行预测
    id<MLFeatureProvider> output = [_model predictionFromFeatures:input error:&error];
    
    if (error) {
        NSLog(@"Prediction error: %@", error);
        return [NSArray array];
    }

    // 获取坐标和置信度
    MLFeatureValue* coordinatesValue = [output featureValueForName:@"coordinates"];
    MLFeatureValue* confidenceValue = [output featureValueForName:@"confidence"];
    
    if (!coordinatesValue || !confidenceValue) {
        std::printf("错误：无法获取坐标或置信度输出\n");
        return [NSArray array];
    }
    
    MLMultiArray* coordinates = coordinatesValue.multiArrayValue;
    MLMultiArray* confidence = confidenceValue.multiArrayValue;
    
    if (!coordinates || !confidence) {
        std::printf("错误：无法获取多数组值\n");
        return [NSArray array];
    }
    
    // 获取类别数量
    NSInteger numClasses = [confidence.shape[1] integerValue];
    if (numClasses <= 0) {
        std::printf("错误：无效的类别数量\n");
        return [NSArray array];
    }
    
    // 创建检测结果数组
    NSMutableArray* results = [NSMutableArray array];
    
    // 获取坐标和置信度值
    float* coords = (float*)coordinates.dataPointer;
    float* confs = (float*)confidence.dataPointer;
    
    if (!coords || !confs) {
        std::printf("错误：无法获取坐标或置信度数据\n");
        return [NSArray array];
    }
    
    // 处理后处理
    DetectionResult result = [self processDetectionResult:coords 
                                              confidence:confs 
                                             imageWidth:originalWidth 
                                            imageHeight:originalHeight
                                             numClasses:numClasses];
    
    // 检查结果是否有效
    if (result.confidence > 0) {
        // 将结果包装为NSValue并添加到数组
        NSValue* value = [NSValue valueWithBytes:&result objCType:@encode(DetectionResult)];
        [results addObject:value];
        return results;
    }

    return nil;
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