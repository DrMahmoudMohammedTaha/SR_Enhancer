import cv2
import numpy as np
from scipy.stats import entropy

def analyze_image_sharpness(image):


    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Input must be a valid OpenCV image (np.ndarray).")

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # 1. Variance of Laplacian (blur detection)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # 2. Tenengrad (based on Sobel gradient magnitude)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_mag = np.sqrt(gx**2 + gy**2)
    tenengrad = np.mean(grad_mag)

    # 3. Histogram entropy
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    hist_entropy = entropy(hist_norm + 1e-7)  # avoid log(0)

    # Optional print
    # print(f"Sharpness Metrics:")
    # print(f"  • Laplacian Variance  : {laplacian_var:.2f}")
    # print(f"  • Tenengrad Gradient  : {tenengrad:.2f}")
    # print(f"  • Histogram Entropy   : {hist_entropy:.4f}")

    return round(laplacian_var, 3), round(tenengrad,3), round(hist_entropy,3)
