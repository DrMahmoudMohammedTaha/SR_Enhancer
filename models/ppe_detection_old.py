from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

model_hardhat = YOLO("models\\ppe_hardhat.pt")
model_avarol = YOLO("models\\ppe_overall.pt")

classNames_hardhat_model = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask',
    'NO-Safety Vest', 'Person', 'Safety Cone',
    'Safety Vest', 'machinery', 'vehicle'
]
classNames_avarol_model = ['avarol', 'no_vest']

# Only these classes will be processed and displayed
TARGET_CLASSES = {
    'Mask': (0, 255, 0),            # Green
    'Safety Vest': (0, 0, 255),     # Red
    'avarol': (255, 0, 0),          # Blue (assuming "avarol" is head cap)
}

def detect_ppe(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:

    detected_items = []
    img_raw = img.copy()

    # First model: masks and vests
    results_hardhat = model_hardhat(img, stream=True)
    for r in results_hardhat:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames_hardhat_model[cls]

            if conf > 0.4 and currentClass in TARGET_CLASSES:
                color = TARGET_CLASSES[currentClass]
                cvzone.putTextRect(
                    img, f'{currentClass}', (max(0, x1), max(40, y1)),
                    scale=2, thickness=2, colorB=color, colorT=(255,255,255), colorR=color, offset=10
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                detected_items.append({
                    "class": currentClass,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

    # Second model: head cap (avarol)
    results_avarol = model_avarol(img, stream=True)
    for r in results_avarol:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames_avarol_model[cls]

            # Only process "avarol" class (head cap)
            if conf > 0.3 and currentClass == "avarol":
                color = TARGET_CLASSES.get("avarol", (255, 0, 0))
                cvzone.putTextRect(
                    img, f'{currentClass}', (max(0, x1), max(40, y1)),
                    scale=2, thickness=2, colorB=color, colorT=(255,255,255), colorR=color, offset=10
                )
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                detected_items.append({
                    "class": currentClass,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

    return img_raw, img, detected_items