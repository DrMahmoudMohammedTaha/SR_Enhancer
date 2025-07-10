import numpy as np
from ultralytics import YOLO
import cv2
import torch
import socket
import time

# PTZ Control Parameters
PTZ_IP = "192.168.100.254"
PTZ_PORT = 30000
DEADZONE = 1  # Pixels where no movement occurs
MAX_SPEED = 0x3F  # Maximum speed value (Pelco-D max is 0x3F)
MIN_SPEED = 0x05  # Minimum speed value

def send_pelco_d_command(pan_dir, tilt_dir, pan_speed, tilt_speed):
    """Send combined Pelco-D command with independent pan/tilt speeds"""
    cmd = [0x00, 0x00, 0x00, 0x00]
    
    # Set pan direction bits
    if pan_dir == "left":
        cmd[1] |= 0x04
    elif pan_dir == "right":
        cmd[1] |= 0x02
    
    # Set tilt direction bits
    if tilt_dir == "up":
        cmd[1] |= 0x08
    elif tilt_dir == "down":
        cmd[1] |= 0x10
    
    # Set speeds (0x00 to 0x3F)
    cmd[2] = min(max(int(pan_speed), 0), 0x3F)
    cmd[3] = min(max(int(tilt_speed), 0), 0x3F)
    
    checksum = (0x01 + cmd[0] + cmd[1] + cmd[2] + cmd[3]) % 256
    message = bytes([0xFF, 0x01, cmd[0], cmd[1], cmd[2], cmd[3], checksum])
    
    # Debugging: Print the Pelco-D command being sent
    print(f"Pelco-D Command Sent: {message.hex().upper()}")

    try:
        ptz_socket.sendto(message, (PTZ_IP, PTZ_PORT))
    except Exception as e:
        print(f"Error sending PTZ command: {e}")

# Verify GPU
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Load YOLO model
model = YOLO("E:/LongRangeCameraProject/Python/Yolov8/datasets/yolo11n.pt")

# RTSP stream setup
rtsp_url = "rtsp://admin:admin_1234@192.168.100.13:554/cam/realmonitor?channel=1&subtype=0&transportMode=TCP"
rtsp_url = 0
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create UDP socket
ptz_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Window setup
window_name = "YOLOv8 + KCF Tracking with PTZ Control"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Tracking variables
frame_skip = 2
frame_count = 0
selected_object = None
tracking = False
tracker = None
last_move_time = 0
move_delay = 0.1  # 100ms between commands

def draw_bullet_and_lines(frame):
    """Draw targeting reticle"""
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), 2)
    cv2.line(frame, (center_x, center_y-25), (center_x, center_y+25), (0, 255, 0), 2)
    cv2.line(frame, (center_x-25, center_y), (center_x+25, center_y), (0, 255, 0), 2)
    cv2.circle(frame, (center_x, center_y), DEADZONE, (0, 0, 255), 1)
    return frame

def draw_status_overlay(frame, tracking_status, ptz_status=""):
    """Draw status information"""
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 140), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(frame, f"Tracking: {'ACTIVE' if tracking_status else 'INACTIVE'}",
               (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (0, 255, 0) if tracking_status else (0, 0, 255), 2)
    cv2.putText(frame, f"PTZ: {ptz_status}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, "Left-click to select target", (20, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Space to stop tracking", (20, 130), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame

def mouse_callback(event, x, y, flags, param):
    global selected_object, tracking, tracker
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = float('inf')
        selected_object = None
        for box in param['boxes']:
            cx = (box[0] + box[2]) // 2
            cy = (box[1] + box[3]) // 2
            dist = (x-cx)**2 + (y-cy)**2
            if dist < min_dist:
                min_dist = dist
                selected_object = box
        if selected_object:
            start_tracking(param['frame'])

def start_tracking(frame):
    global tracker, tracking
    tracker = cv2.TrackerKCF_create()
    x1, y1, x2, y2 = selected_object
    tracker.init(frame, (x1, y1, x2-x1, y2-y1))
    tracking = True
    print("Tracking started")

def calculate_pan_tilt_speed(diff_x, diff_y):
    """Calculate movement directions and speeds"""
    current_time = time.time()
    global last_move_time
    
    if current_time - last_move_time < move_delay:
        return "stop", "stop", 0, 0
    
    last_move_time = current_time
    distance = np.sqrt(diff_x**2 + diff_y**2)
    
    if distance > DEADZONE:
        norm_speed = min(distance / (width/2), 1.0)
        base_speed = MIN_SPEED + (MAX_SPEED - MIN_SPEED) * norm_speed
        
        pan_dir = "right" if diff_x > 0 else "left" if diff_x < -DEADZONE else "stop"
        tilt_dir = "down" if diff_y > 0 else "up" if diff_y < -DEADZONE else "stop"
        
        if pan_dir != "stop" and tilt_dir != "stop":
            ratio = abs(diff_x) / (abs(diff_x) + abs(diff_y))
            pan_speed = base_speed * ratio
            tilt_speed = base_speed * (1 - ratio)
        elif pan_dir != "stop":
            pan_speed = base_speed
            tilt_speed = 0
        elif tilt_dir != "stop":
            pan_speed = 0
            tilt_speed = base_speed
        else:
            return "stop", "stop", 0, 0
        
        return pan_dir, tilt_dir, pan_speed, tilt_speed
    
    return "stop", "stop", 0, 0

# Initialize mouse callback
cv2.setMouseCallback(window_name, mouse_callback, {'boxes': [], 'frame': None})

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # YOLO detection
    results = model.predict(source=frame, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()
    
    # Get detection boxes
    boxes = []
    for box in results[0].boxes.xyxy:
        boxes.append(tuple(map(int, box)))
    
    cv2.setMouseCallback(window_name, mouse_callback, {'boxes': boxes, 'frame': annotated_frame.copy()})

    # Tracking logic
    ptz_status = "Idle"
    if tracking and tracker:
        success, bbox = tracker.update(annotated_frame)
        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            
            target_x, target_y = x + w//2, y + h//2
            diff_x, diff_y = target_x - width//2, target_y - height//2
            
            cv2.circle(annotated_frame, (target_x, target_y), 5, (0, 255, 255), -1)
            cv2.line(annotated_frame, (width//2, height//2), (target_x, target_y), (0, 255, 255), 2)
            
            pan_dir, tilt_dir, pan_speed, tilt_speed = calculate_pan_tilt_speed(diff_x, diff_y)
            
            if pan_dir != "stop" or tilt_dir != "stop":
                send_pelco_d_command(pan_dir, tilt_dir, pan_speed, tilt_speed)
                ptz_status = f"Pan: {pan_dir}@{pan_speed} Tilt: {tilt_dir}@{tilt_speed}"
            else:
                send_pelco_d_command("stop", "stop", 0, 0)
                ptz_status = "Centered"
            
            cv2.putText(annotated_frame, f"X: {diff_x}, Y: {diff_y}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(annotated_frame, f"Speed: {max(pan_speed, tilt_speed)}", 
                       (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            tracking = False
            send_pelco_d_command("stop", "stop", 0, 0)
            ptz_status = "Tracking lost"

    # Draw UI and display
    annotated_frame = draw_bullet_and_lines(annotated_frame)
    annotated_frame = draw_status_overlay(annotated_frame, tracking, ptz_status)
    cv2.imshow(window_name, annotated_frame)
    
    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        tracking = False
        send_pelco_d_command("stop", "stop", 0, 0)

cap.release()
ptz_socket.close()
cv2.destroyAllWindows()
