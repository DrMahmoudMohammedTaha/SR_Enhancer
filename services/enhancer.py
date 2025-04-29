import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def get_center(x, y, w, h):
    """Calculate center of an object"""
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy


# def calculate_metrics(original, enhanced):
#     original_np = (original.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype('uint8')
#     enhanced_np = (enhanced.cpu().numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype('uint8')
#     psnr_value = psnr(original_np, enhanced_np, data_range=255)
#     ssim_value = ssim(original_np, enhanced_np, data_range=255, channel_axis=2)
#     return psnr_value, ssim_value

# def compute_psnr_ssim(original_frame, enhanced_frame, device):
#     original = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
#     enhanced = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
#     original_tensor = transforms.ToTensor()(Image.fromarray(original)).to(device)
#     enhanced_tensor = transforms.ToTensor()(Image.fromarray(enhanced)).to(device)
#     psnr_val, ssim_val = calculate_metrics(original_tensor, enhanced_tensor)
#     return psnr_val, ssim_val


# def compute_psnr_ssim(original_frame, enhanced_frame):
#     original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
#     enhanced_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
#     psnr_val = psnr(original_rgb, enhanced_rgb, data_range=255)
#     ssim_val = ssim(original_rgb, enhanced_rgb, data_range=255, channel_axis=2)
    
#     return psnr_val, ssim_val


executor = ThreadPoolExecutor(max_workers=2)  # Run PSNR and SSIM concurrently

def compute_psnr(original_rgb, enhanced_rgb):
    return psnr(original_rgb, enhanced_rgb, data_range=255)

def compute_ssim(original_rgb, enhanced_rgb):
    return ssim(original_rgb, enhanced_rgb, data_range=255, channel_axis=2)

def compute_psnr_ssim(original_frame, enhanced_frame):
    original_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
    future_psnr = executor.submit(compute_psnr, original_rgb, enhanced_rgb)
    future_ssim = executor.submit(compute_ssim, original_rgb, enhanced_rgb)
    
    psnr_val = future_psnr.result()
    ssim_val = future_ssim.result()
    
    return psnr_val, ssim_val

def auto_tone_image(img):
    result = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        min_val, max_val = ch.min(), ch.max()
        if max_val > min_val:
            stretched = (ch - min_val) * (255.0 / (max_val - min_val))
            result[:, :, c] = np.clip(stretched, 0, 255)
        else:
            result[:, :, c] = ch
    return result.astype(np.uint8)


def apply_sharpening(frame, laplacian, sobelx, sobely):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = frame.copy()

    if laplacian:
        lap = np.uint8(np.absolute(cv2.Laplacian(gray, cv2.CV_64F)))
        for i in range(3):
            result[:, :, i] = cv2.addWeighted(result[:, :, i], 1, lap, 0.5, 0)

    if sobelx:
        sobelX = np.uint8(np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0)))
        for i in range(3):
            result[:, :, i] = cv2.addWeighted(result[:, :, i], 1, sobelX, 0.5, 0)

    if sobely:
        sobelY = np.uint8(np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1)))
        for i in range(3):
            result[:, :, i] = cv2.addWeighted(result[:, :, i], 1, sobelY, 0.5, 0)

    return result