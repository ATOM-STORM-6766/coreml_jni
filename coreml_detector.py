import cv2
import sys
import numpy as np
import coremltools
from PIL import Image
from typing import List, Tuple

class ObjDetectObservation:
    def __init__(self, obj_class: int, confidence: float, corners: np.ndarray):
        self.obj_class = obj_class
        self.confidence = confidence
        self.corners = corners

class ObjectDetector:
    def __init__(self):
        self._model = None

    def detect(self, image: np.ndarray, model_path: str) -> List[ObjDetectObservation]:
        # 加载 CoreML 模型
        if self._model is None:
            print("正在加载目标检测模型...")
            self._model = coremltools.models.MLModel(model_path)
            print("模型加载完成")

        # 记录原始图像尺寸
        print(f"原始图像尺寸: {image.shape}")
        
        # 创建缩放后的图像用于模型输入
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image_scaled = np.zeros((640, 640, 3), dtype=np.uint8)
        scaled_height = int(640 / (image.shape[1] / image.shape[0]))
        bar_height = int((640 - scaled_height) / 2)
        print(f"缩放后图像尺寸: {image_scaled.shape}")
        print(f"缩放高度: {scaled_height}, 上下填充高度: {bar_height}")
        
        image_scaled[bar_height:bar_height + scaled_height, 0:640] = cv2.resize(image, (640, scaled_height))

        # 运行 CoreML 模型
        image_coreml = Image.fromarray(image_scaled)
        prediction = self._model.predict({"image": image_coreml})

        observations: List[ObjDetectObservation] = []
        for coordinates, confidence in zip(prediction["coordinates"], prediction["confidence"]):
            obj_class = max(range(len(confidence)), key=confidence.__getitem__)
            confidence = float(confidence[obj_class])
            x = coordinates[0] * image.shape[1]
            y = ((coordinates[1] * 640 - bar_height) / scaled_height) * image.shape[0]
            width = coordinates[2] * image.shape[1]
            height = coordinates[3] / (scaled_height / 640) * image.shape[0]

            print(f"检测到目标 - 类别: {obj_class}, 置信度: {confidence:.2f}")
            print(f"原始坐标: x={coordinates[0]:.2f}, y={coordinates[1]:.2f}, w={coordinates[2]:.2f}, h={coordinates[3]:.2f}")
            print(f"映射后坐标: x={x:.2f}, y={y:.2f}, w={width:.2f}, h={height:.2f}")

            corners = np.array([
                [x - width/2, y - height/2],
                [x + width/2, y - height/2],
                [x - width/2, y + height/2],
                [x + width/2, y + height/2]
            ])
            
            observations.append(ObjDetectObservation(obj_class, confidence, corners))

        return observations

def draw_detections(image: np.ndarray, observations: List[ObjDetectObservation]) -> np.ndarray:
    img = image.copy()
    for obs in observations:
        # 获取边界框的四个角点
        corners = obs.corners.astype(int)
        # 绘制边界框
        cv2.rectangle(img, 
                     (corners[0][0], corners[0][1]), 
                     (corners[3][0], corners[3][1]), 
                     (0, 255, 0), 2)
        # 添加类别和置信度标签
        label = f"Class: {obs.obj_class}, Conf: {obs.confidence:.2f}"
        cv2.putText(img, label, (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def main():
    if len(sys.argv) != 3:
        print("用法: python draw_box.py <图片路径> <模型路径>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2]
    
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"错误：无法读取图片 {image_path}")
        return
    
    # 创建检测器并运行检测
    detector = ObjectDetector()
    observations = detector.detect(image, model_path)
    
    # 绘制检测结果
    result_image = draw_detections(image, observations)
    
    # 显示结果
    cv2.imshow('Detection Results', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果
    output_path = 'output_' + image_path
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")

if __name__ == "__main__":
    main() 
