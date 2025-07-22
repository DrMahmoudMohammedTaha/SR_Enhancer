from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np

model_hardhat = YOLO("models\\ppe_hardhat.pt")
model_avarol = YOLO("models\\ppe_overall.pt")
classNames_hardhat_model = [
    'Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest',
    'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle'
]
classNames_avarol_model = ['avarol', 'no_vest']

def detect_ppe(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:


    detected_items = []
    results_hardhat = model_hardhat(img, stream=True)
    results_avarol =  model_avarol(img, stream=True)
    img_raw = img.copy()
    for r in results_hardhat:
        boxes = r.boxes
        for box in boxes:
            x_left, y1_top, x2_right, y2_bottom = map(int, box.xyxy[0])
            print(x_left, y1_top, x2_right, y2_bottom)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames_hardhat_model[cls]
            print(currentClass)
            if conf > 0.4 and currentClass not in [
                'Hardhat', 'Mask', 'Safety Cone', 'Safety Vest',
                'machinery', 'vehicle', 'NO-Safety Vest'
            ]:
                if currentClass in ['NO-Hardhat', "NO-Mask"]:
                    myColor = (0, 0, 255)
                elif currentClass == 'Person':
                    myColor = (255, 0, 0)
                
                cvzone.putTextRect(
                    img, f'{classNames_hardhat_model[cls]}', (max(0, x_left), max(40, y1_top)),
                    scale=2, thickness=2, colorB=myColor, colorT=(255,255,255), colorR=myColor, offset=10
                )
                cv2.rectangle(img, (x_left, y1_top), (x2_right, y2_bottom), myColor, 3)
                detected_items.append((currentClass, conf, (x_left, y1_top, x2_right, y2_bottom)))
    
    for r in results_avarol:
        boxes = r.boxes
        for box in boxes:
            x_left, y1_top, x2_right, y2_bottom = map(int, box.xyxy[0])
            print(x_left, y1_top, x2_right, y2_bottom)
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames_avarol_model[cls]
            print(currentClass)
            if conf > 0.3 and currentClass in ["avarol", "no_vest"]:
                myColor = (0, 0, 255)
                cvzone.putTextRect(
                    img, f'{classNames_avarol_model[cls]}', (max(0, x_left), max(40, y1_top)),
                    scale=2, thickness=2, colorB=myColor, colorT=(255,255,255), colorR=myColor, offset=10
                )
                cv2.rectangle(img, (x_left, y1_top), (x2_right, y2_bottom), myColor, 3)
                detected_items.append((currentClass, conf, (x_left, y1_top, x2_right, y2_bottom)))
    
    return img_raw, img, detected_items