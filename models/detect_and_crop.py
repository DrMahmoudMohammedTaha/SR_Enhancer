import os
import torch
import cv2
from ultralytics import YOLO

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to YOLO model
YOLO_MODEL_PATH = os.path.join('C:\\Users\\Mahmoud_Taha\\Downloads\\best.pt')

# Input and output directories
INPUT_DIR = 'D:\\_research\\pedestrian_datasets\\++ far car - VAID.v1i.yolov8\\valid\\images'
OUTPUT_DIR = os.path.join(INPUT_DIR, 'cropped')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_yolo_model(yolo_path: str):
    try:
        model = YOLO(yolo_path)
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {str(e)}")
        return None

def predict(model, image, conf_threshold=0.25):
    results = model.predict(image, conf=conf_threshold, iou=0.5)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf)
            cls = int(box.cls)
            if cls == 1:  # Assuming class 0 is the target class
                detections.append((x1, y1, x2, y2, conf, cls))
    return detections

def process_images():
    model = load_yolo_model(YOLO_MODEL_PATH)
    if model is None:
        print("Failed to load YOLO model. Exiting.")
        return

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(INPUT_DIR, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
            detections = predict(model, image)
            for idx, (x1, y1, x2, y2, conf, cls) in enumerate(detections):
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                # Save with original name (if multiple detections, add index)
                if len(detections) > 1:
                    save_name = f"{os.path.splitext(filename)[0]}_{idx}{os.path.splitext(filename)[1]}"
                else:
                    save_name = filename
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, crop)
                print(f"Saved cropped object to {save_path}")

if __name__ == "__main__":
    process_images() 