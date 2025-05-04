import cv2
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()

    def track_objects(self, frame):
        """Apply object tracking to the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = self.bg_subtractor.apply(gray)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 20:  # Filter based on size
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return frame