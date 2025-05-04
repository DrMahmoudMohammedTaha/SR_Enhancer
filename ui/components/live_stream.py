import cv2
import time

class LiveStreamHandler:
    def __init__(self):
        self.cap = None

    def start_stream(self, rtsp_url):
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise ValueError("Could not open RTSP stream.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)  # Retry after delay
                continue
            
            # Handle the frame (e.g., display or process)
            pass