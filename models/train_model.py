

from ultralytics import YOLO

# ✅ Load a pretrained YOLOv8 model
model = YOLO("yolov8m.pt")  # or your own checkpoint

# ✅ Train the model with saving after every epoch
results = model.train(
    data="C:\\Users\\Mahmoud_Taha\\Downloads\\++far_object_dataset\\data.yaml",  # path to your YAML file
    epochs=5,
    imgsz=640,
    batch=16,
    name="yolov8_custom_far_objects",  # this creates runs/detect/yolov8_custom_far_objects
    save=True,
    save_period=1  # <-- Save model after each epoch
)

# ✅ (Optional) Save final model explicitly
model.save("C:\\Users\\Mahmoud_Taha\\Downloads\\++far_object_dataset\\yolov8_custom_far_objects_final.pt")
