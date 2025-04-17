#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct DetectionResult {
    float x1, y1, x2, y2;  // 边界框坐标
    float confidence;      // 置信度
    int class_id;         // 类别ID
};

class CoreMLDetector {
public:
    CoreMLDetector(const std::string& modelPath);
    ~CoreMLDetector();
    
    void setCoreMask(int coreMask);
    std::vector<DetectionResult> detect(const cv::Mat& image, double nmsThresh, double boxThresh);
    
private:
    void* impl_;  // 指向 Objective-C 实现的指针
}; 