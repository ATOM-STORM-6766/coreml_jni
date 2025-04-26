#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct DetectionResult {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float confidence;      // Confidence
    int class_id;         // Class ID
};

class CoreMLDetector {
public:
    explicit CoreMLDetector(const std::string& modelPath);
    ~CoreMLDetector();
    
    int setCoreMask(int coreMask);
    std::vector<DetectionResult> detect(const cv::Mat& image, double nmsThresh, double boxThresh);
    
private:
    void* impl_;  // Pointer to Objective-C implementation
}; 