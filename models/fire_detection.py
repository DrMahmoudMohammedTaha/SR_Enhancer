
from ultralytics import YOLO
import cv2
import math

class FireSmokeDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classNames = ["fire", "smoke"]

    def detect_and_draw(self, frame):
        results = self.model(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                if 0 <= cls < len(self.classNames):
                    class_name = self.classNames[cls]
                else:
                    print(f"Warning: Detected class index {cls} outside expected range!")
                    continue

                color = (0, 0, 255) if class_name == "fire" else (128, 128, 128)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                confidence = math.ceil((box.conf[0] * 100)) / 100
                label = f"{class_name}: {confidence:.2f}"
                org = (x1, y1 - 10)
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                thickness = 2
                cv2.putText(frame, label, org, font, fontScale, color, thickness)
        return frame