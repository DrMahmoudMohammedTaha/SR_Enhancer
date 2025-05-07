
import torch
import cv2

yolo_model_loaded = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_path = "models\\yolo_model.pt"
yolo_model = None

from ultralytics import YOLO

def load_yolo_model(yolo_path: str):
    global yolo_model
    global yolo_model_loaded

    try:
        yolo_model = YOLO(yolo_path)
        yolo_model_loaded = True
    except Exception as e:
        print(f"Error loading yolo_model: {str(e)}")
        yolo_model_loaded = False

    return yolo_model_loaded

def predict(image, conf_threshold=0.25):
    global yolo_model

    results = yolo_model.predict(image, conf=conf_threshold, iou=0.5)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf)
            cls = int(box.cls)
            print(f"Detected: {cls} with confidence: {conf}")
            detections.append((x1, y1, x2, y2, conf, cls))
    return detections


def apply_yolo_model(image, conf_threshold=0.25):
    
    detections = predict(image, conf_threshold=conf_threshold)

    
    for x1, y1, x2, y2, conf, cls in detections:
        # Draw bounding box
    
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

        # Prepare label
        label = f'{cls}: {conf:.2f}'
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw label background
        cv2.rectangle(image, (x1, y1 - label_height - baseline), (x1 + label_width, y1), (0, 255, 0), -1)

        # Put label text
        cv2.putText(image, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=1)

    return image
