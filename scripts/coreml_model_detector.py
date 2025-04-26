import cv2
import sys
import numpy as np
import coremltools
from PIL import Image
from typing import List, Tuple
import time
import threading
import queue
import coremltools.proto.FeatureTypes_pb2 as ft
import platform

class PerformanceMetrics:
    def __init__(self):
        self.frame_read_time = 0
        self.preprocess_time = 0
        self.model_inference_time = 0
        self.postprocess_time = 0
        self.draw_time = 0
        self.total_time = 0
        self.frame_count = 0

    def update(self, read_time, preprocess_time, inference_time, postprocess_time, draw_time):
        self.frame_read_time += read_time
        self.preprocess_time += preprocess_time
        self.model_inference_time += inference_time
        self.postprocess_time += postprocess_time
        self.draw_time += draw_time
        self.total_time += read_time + preprocess_time + inference_time + postprocess_time + draw_time
        self.frame_count += 1

    def get_averages(self):
        if self.frame_count == 0:
            return 0, 0, 0, 0, 0, 0
        return (
            self.frame_read_time / self.frame_count * 1000,
            self.preprocess_time / self.frame_count * 1000,
            self.model_inference_time / self.frame_count * 1000,
            self.postprocess_time / self.frame_count * 1000,
            self.draw_time / self.frame_count * 1000,
            self.total_time / self.frame_count * 1000
        )

    def reset(self):
        self.frame_read_time = 0
        self.preprocess_time = 0
        self.model_inference_time = 0
        self.postprocess_time = 0
        self.draw_time = 0
        self.total_time = 0
        self.frame_count = 0

class ObjDetectObservation:
    def __init__(self, obj_class: int, confidence: float, corners: np.ndarray):
        self.obj_class = obj_class
        self.confidence = confidence
        self.corners = corners

class FrameReader:
    def __init__(self, camera_id: int):
        self._camera_id = camera_id
        self._cap = cv2.VideoCapture(camera_id)
        self._frame_queue = queue.Queue(maxsize=1)
        self._metrics = PerformanceMetrics()
        self._thread = threading.Thread(target=self._read_thread, daemon=True)
        self._running = True
        self._thread.start()

    def _read_thread(self):
        while self._running:
            try:
                read_start = time.time()
                ret, frame = self._cap.read()
                read_time = time.time() - read_start

                if not ret:
                    print("Error: Failed to read video frame")
                    continue

                # Preprocess image
                preprocess_start = time.time()
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                image_scaled = np.zeros((640, 640, 3), dtype=np.uint8)
                scaled_height = int(640 / (frame.shape[1] / frame.shape[0]))
                bar_height = int((640 - scaled_height) / 2)
                image_scaled[bar_height:bar_height + scaled_height, 0:640] = cv2.resize(frame, (640, scaled_height))
                preprocess_time = time.time() - preprocess_start

                # Update performance metrics
                self._metrics.update(read_time, preprocess_time, 0, 0, 0)

                # Put processed frame into queue
                if not self._frame_queue.empty():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put((frame, image_scaled, bar_height, scaled_height))
            except Exception as e:
                print(f"Frame reading thread error: {e}")

    def get_frame(self):
        try:
            return self._frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None, None, None, None

    def get_metrics(self):
        return self._metrics

    def stop(self):
        self._running = False
        self._thread.join()
        self._cap.release()

class ObjectDetector:
    def __init__(self, model_path: str):
        print("Loading target detection model...")
        if platform.system() == 'Darwin':
            self._compute_units = coremltools.ComputeUnit.ALL
        else:
            self._compute_units = coremltools.ComputeUnit.CPU_ONLY
            
        self._model = coremltools.models.MLModel(model_path, compute_units=self._compute_units)
        self._metrics = PerformanceMetrics()
        self._thread = threading.Thread(target=self._detection_thread, daemon=True)
        self._input_queue = queue.Queue(maxsize=1)
        self._output_queue = queue.Queue(maxsize=1)
        self._running = True
        self._thread.start()

    def _detection_thread(self):
        while self._running:
            try:
                frame_data = self._input_queue.get()
                if frame_data is None:
                    break
                
                frame, image_scaled, bar_height, scaled_height = frame_data
                
                # Run model
                inference_start = time.time()
                image_coreml = Image.fromarray(image_scaled)
                prediction = self._model.predict({"image": image_coreml})
                inference_time = time.time() - inference_start

                # Process results
                postprocess_start = time.time()
                observations: List[ObjDetectObservation] = []
                for coordinates, confidence in zip(prediction["coordinates"], prediction["confidence"]):
                    obj_class = max(range(len(confidence)), key=confidence.__getitem__)
                    confidence = float(confidence[obj_class])
                    x = coordinates[0] * frame.shape[1]
                    y = ((coordinates[1] * 640 - bar_height) / scaled_height) * frame.shape[0]
                    width = coordinates[2] * frame.shape[1]
                    height = coordinates[3] / (scaled_height / 640) * frame.shape[0]
                    corners = np.array([
                        [x - width/2, y - height/2],
                        [x + width/2, y - height/2],
                        [x - width/2, y + height/2],
                        [x + width/2, y + height/2]
                    ])
                    observations.append(ObjDetectObservation(obj_class, confidence, corners))
                postprocess_time = time.time() - postprocess_start

                # Update performance metrics
                self._metrics.update(0, 0, inference_time, postprocess_time, 0)
                
                self._output_queue.put((frame, observations))
            except Exception as e:
                print(f"Detection thread error: {e}")
                self._output_queue.put((None, []))

    def detect(self, frame_data):
        try:
            # Clear old data from queue
            while not self._input_queue.empty():
                self._input_queue.get_nowait()
            while not self._output_queue.empty():
                self._output_queue.get_nowait()
            
            # Send new image
            self._input_queue.put(frame_data)
            
            # Get results
            return self._output_queue.get(timeout=1.0)
        except queue.Empty:
            return None, []
        except Exception as e:
            print(f"Detection error: {e}")
            return None, []

    def get_metrics(self):
        return self._metrics

    def stop(self):
        self._running = False
        self._input_queue.put(None)
        self._thread.join()

def draw_detections(image: np.ndarray, observations: List[ObjDetectObservation]) -> np.ndarray:
    draw_start = time.time()
    img = image.copy()
    for obs in observations:
        corners = obs.corners.astype(int)
        cv2.rectangle(img, 
                     (corners[0][0], corners[0][1]), 
                     (corners[3][0], corners[3][1]), 
                     (0, 255, 0), 2)
        label = f"Class: {obs.obj_class}, Conf: {obs.confidence:.2f}"
        cv2.putText(img, label, (corners[0][0], corners[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    draw_time = time.time() - draw_start
    return img, draw_time

def list_available_cameras():
    """List all available cameras"""
    available_cameras = []
    for i in range(10):  # Check first 10 possible camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def realtime_detection(model_path: str):
    cameras = list_available_cameras()
    if not cameras:
        print("Error: No available cameras found")
        return
    
    print("\nAvailable cameras:")
    for cam_id in cameras:
        print(f"Camera {cam_id}")
    
    while True:
        try:
            selected_cam = int(input("\nPlease enter the camera number to use: "))
            if selected_cam in cameras:
                break
            else:
                print("Invalid camera number, please try again")
        except ValueError:
            print("Please enter a valid number")
    
    # Create frame reader and detector
    frame_reader = FrameReader(selected_cam)
    detector = ObjectDetector(model_path)
    frame_metrics = frame_reader.get_metrics()
    detector_metrics = detector.get_metrics()
    
    # Performance statistics
    fps = 0
    frame_count = 0
    start_time = time.time()
    last_frame_time = time.time()
    
    print("\nStarting real-time detection...")
    print("Press 'q' to exit")
    
    while True:
        # Get frame
        frame, image_scaled, bar_height, scaled_height = frame_reader.get_frame()
        if frame is None:
            continue
        
        # Perform detection
        result_frame, observations = detector.detect((frame, image_scaled, bar_height, scaled_height))
        if result_frame is None:
            continue
        
        # Draw results
        result_frame, draw_time = draw_detections(result_frame, observations)
        
        # Calculate actual FPS
        current_time = time.time()
        frame_interval = current_time - last_frame_time
        last_frame_time = current_time
        
        # Update FPS display
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            frame_count = 0
            start_time = time.time()
            
            # Get average time
            read_avg, preprocess_avg, _, _, _, _ = frame_metrics.get_averages()
            _, _, inference_avg, postprocess_avg, _, _ = detector_metrics.get_averages()
            print(f"\nPerformance Analysis (ms):")
            print(f"Frame Reading: {read_avg:.1f}")
            print(f"Preprocessing: {preprocess_avg:.1f}")
            print(f"Model Inference: {inference_avg:.1f}")
            print(f"Postprocessing: {postprocess_avg:.1f}")
            print(f"Drawing: {draw_time*1000:.1f}")
            print(f"Total Time: {frame_interval*1000:.1f}")
            frame_metrics.reset()
            detector_metrics.reset()
        
        # Display FPS and frame interval
        cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Frame Interval: {frame_interval*1000:.1f}ms", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display results
        cv2.imshow('Real-time Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    frame_reader.stop()
    detector.stop()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 3:
        print("Usage: python coreml_detector.py <image_path/camera> <model_path>")
        print("Use 'camera' as image path to use camera input")
        sys.exit(1)
    
    input_path = sys.argv[1]
    model_path = sys.argv[2]
    
    if input_path.lower() == 'camera':
        realtime_detection(model_path)
    else:
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Failed to read image {input_path}")
            return
        
        detector = ObjectDetector(model_path)
        observations, preprocess_time, inference_time, postprocess_time = detector.detect(image)
        
        result_image, draw_time = draw_detections(image, observations)
        
        print("\nPerformance Analysis (ms):")
        print(f"Preprocessing: {preprocess_time*1000:.1f}")
        print(f"Model Inference: {inference_time*1000:.1f}")
        print(f"Postprocessing: {postprocess_time*1000:.1f}")
        print(f"Drawing: {draw_time*1000:.1f}")
        print(f"Total Time: {(preprocess_time + inference_time + postprocess_time + draw_time)*1000:.1f}")
        
        cv2.imshow('Detection Results', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        output_path = 'output_' + input_path
        cv2.imwrite(output_path, result_image)
        print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 
