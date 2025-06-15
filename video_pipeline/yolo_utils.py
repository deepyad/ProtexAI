# video_pipeline/yolo_utils.py
from ultralytics import YOLO
import numpy as np
import cv2

def load_detection_model(config):
    """Load YOLO model with configuration"""
    model_path = config.get('model', {}).get('path', 'yolov8n.pt')
    confidence = config.get('model', {}).get('confidence', 0.5)
    model = YOLO(model_path)
    model.conf = confidence
    return model


def detect_objects(frame, model):
    """Run inference with proper error handling"""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return []

    try:
        # Explicitly convert to float32 and BGR2RGB
        frame = cv2.cvtColor(frame.astype('float32'), cv2.COLOR_BGR2RGB)
        
        # Use official prediction interface
        results = model.predict(frame, verbose=False, save=False)
        
        detections = []
        for result in results:
            # Check boxes using numpy-aware logic
            if result.boxes is not None and result.boxes.shape[0] > 0:
                names = result.names
                for box in result.boxes:
                    class_id = int(box.cls)
                    detections.append({
                        "class": names[class_id],
                        "confidence": float(box.conf),
                        "bbox": [float(x) for x in box.xyxy[0].cpu().numpy()]
                    })
        return detections
    except Exception as e:
        print(f"Detection error: {e}")
        return []


