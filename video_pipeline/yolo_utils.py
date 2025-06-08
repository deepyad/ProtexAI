from ultralytics import YOLO

def load_detection_model(config):
    """
    Loads a YOLOv8 model using the path and confidence threshold from the config dictionary.
    """
    model_path = config.get('model', {}).get('path', 'yolov8n.pt')
    confidence = config.get('model', {}).get('confidence', 0.5)
    model = YOLO(model_path)
    model.conf = confidence  # Set confidence threshold for inference
    return model

def detect_objects(frame, model):
    """
    Runs YOLOv8 inference on a frame and returns a list of detections.
    Each detection is a dictionary with class name, confidence, and bounding box.
    """
    results = model(frame, verbose=False)  # Run inference
    detections = []
    for result in results:
        names = result.names  # Dictionary mapping class indices to names
        for box in result.boxes:
            class_id = int(box.cls)
            detections.append({
                "class": names[class_id],
                "confidence": float(box.conf),
                "bbox": [float(x) for x in box.xyxy[0].tolist()]  # [x1, y1, x2, y2]
            })
    return detections
