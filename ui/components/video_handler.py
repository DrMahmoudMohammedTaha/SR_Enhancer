import cv2

class VideoHandler:
    def __init__(self):
        self.cap = None
        self.frame_count = 0
        self.current_frame_number = 0

    def load_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Could not open video file.")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame_number += 1
                return frame
        return None

    def reset_video(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_number = 0