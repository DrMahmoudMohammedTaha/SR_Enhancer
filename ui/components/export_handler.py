import cv2

class ExportHandler:
    def __init__(self, root):
        self.root = root

    def export_video(self, video_handler, output_path):
        """Export video to the specified path"""
        cap = video_handler.cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        out.release()