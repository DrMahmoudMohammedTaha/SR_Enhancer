from ultralytics import YOLO
import cv2
import torch

# Verify GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Load the YOLOv8n model (automatically uses GPU if available)
model = YOLO("E:/LongRangeCameraProject/Python/Yolov8/datasets/yolo11n.pt")

# RTSP stream URL
rtsp_url = "rtsp://admin:admin_1234@192.168.100.13:554/cam/realmonitor?channel=1&subtype=0&transportMode=TCP"
rtsp_url = 0
# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Video properties
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# Output stream setup (replace with your RTSP server URL)
output_path = "rtsp://<output-stream-url>"
fourcc = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Fullscreen window
cv2.namedWindow("YOLOv11n Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv11n Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

frame_skip = 2
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Perform detection (YOLO automatically uses GPU)
    results = model.predict(source=frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Display and stream output
    cv2.imshow("YOLOv11n Detection", annotated_frame)
    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()