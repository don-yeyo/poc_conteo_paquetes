import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2


class ObjectDetector:
    def __init__(self, model_path=None, max_results=10, score_threshold=0.5):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            max_results=max_results,
            score_threshold=score_threshold
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        self.mp_detections = None
    
    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        self.mp_detections = self.detector.detect_for_video(mp_image, timestamp_ms=int(cv2.getTickCount()))
        return self.mp_detections
    
    def get_detections(self):
        return self.mp_detections


def create_default_detector():
    base_options = python.BaseOptions(model_asset_path='efficientdet_lite0.tflite')
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        max_results=10,
        score_threshold=0.5
    )
    return vision.ObjectDetector.create_from_options(options)


def load_detector(model_path='efficientdet_lite0.tflite'):
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            max_results=10,
            score_threshold=0.3
        )
        return vision.ObjectDetector.create_from_options(options)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def detect_objects(detector, frame, timestamp_ms):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    return detector.detect_for_video(mp_image, timestamp_ms)


def draw_detection(frame, detection, color=(0, 255, 0), thickness=2):
    if not detection or not detection.detections:
        return frame
    
    height, width = frame.shape[:2]
    
    for det in detection.detections:
        bbox = det.bounding_box
        x = int(bbox.origin_x)
        y = int(bbox.origin_y)
        w = int(bbox.width)
        h = int(bbox.height)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        
        if det.categories:
            category = det.categories[0]
            label = f"{category.category_name}: {category.score:.2f}"
            cv2.putText(frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame