import cv2
import numpy as np

class FrameEnhancer:
    def enhance_frame(self, frame, contrast=1.0, brightness=0, gamma=1.0):
        """Enhance frame with contrast, brightness, and gamma adjustments"""
        adjusted = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
            adjusted = cv2.LUT(adjusted, table)
        return adjusted