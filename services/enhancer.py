import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from torchvision import transforms
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from skimage import restoration, io, img_as_ubyte
from skimage.color import rgb2gray

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



def unsharp_mask(image, sigma=1.0, strength=1.5):
    # Blur the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    # Create the mask
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened


# def deblur_image(image):
#     gray = rgb2gray(image)
#     psf = np.ones((5, 5)) / 25
#     deconvolved = restoration.richardson_lucy(gray, psf, num_iter=30)
#     return deconvolved


def deblur_image(image, *, to_uint8=True):

    # Step 1: convert to grayscale in [0, 1] float
    gray = rgb2gray(image).astype(np.float32)

    # Step 2: build a simple 5×5 box PSF
    psf = np.ones((5, 5), dtype=np.float32) / 25.0

    # Step 3: Richardson–Lucy deconvolution (30 iterations)
    deconvolved = restoration.richardson_lucy(gray, psf, num_iter=30).astype(np.float32)

    if to_uint8:
        # Map from [0, 1] float → [0, 255] uint8
        deconvolved = np.clip(deconvolved * 255, 0, 255).astype(np.uint8)
    else:
        # Keep float but ensure dtype is float32 (accepted by cv2)
        deconvolved = np.clip(deconvolved, 0.0, 1.0).astype(np.float32)

    return deconvolved














import torch
import torch.nn.functional as F
import numpy as np
from skimage.color import rgb2gray

# ---- helper --------------------------------------------------------------
def _psf_to_kernel(psf: np.ndarray, device):
    """
    Turn a 2‑D PSF (numpy) into a depthwise‑conv kernel (torch) with shape
    (1, 1, kH, kW).  Normalised so that its sum is 1 and moved to `device`.
    """
    psf = torch.from_numpy(psf.astype(np.float32))
    psf = psf / psf.sum()
    return psf.to(device).unsqueeze(0).unsqueeze(0)          # (1,1,kH,kW)

# -------------------------------------------------------------------------
def deblur_image_gpu(image: np.ndarray,
                     num_iter: int = 30,
                     psf_size: int = 5,
                     to_uint8: bool = True,
                     device: str = "cuda"):
    """
    Deblur an RGB image on the GPU using Richardson–Lucy.

    Parameters
    ----------
    image     : ndarray (H,W,3) uint8 or float
    num_iter  : iterations of the RL algorithm
    psf_size  : size of the box PSF (odd int)
    to_uint8  : return uint8 (True) or float32 in [0,1] (False)
    device    : 'cuda' (default) or 'cpu'

    Returns
    -------
    deconvolved : ndarray, dtype uint8 or float32
    """
    # ------------------------------------------------------------------
    # 0. to grayscale, float32 on CUDA
    # ------------------------------------------------------------------
    gray = rgb2gray(image).astype(np.float32)
    img_t = torch.from_numpy(gray).to(device)       # (H,W)
    img_t = img_t.unsqueeze(0).unsqueeze(0)         # (1,1,H,W)

    # ------------------------------------------------------------------
    # 1. prepare PSF + flipped PSF for conv / correlation
    # ------------------------------------------------------------------
    psf        = np.ones((psf_size, psf_size), dtype=np.float32)
    kernel     = _psf_to_kernel(psf, device)                      # (1,1,k,k)
    kernel_flip = torch.flip(kernel, dims=[-1, -2])               # rotated 180°

    # ------------------------------------------------------------------
    # 2. RL iterations: GPU‑accelerated
    # ------------------------------------------------------------------
    estimate = img_t.clone() + 1e-6                               # avoid /0
    for _ in range(num_iter):
        # forward projection
        conv_est = F.conv2d(estimate, kernel, padding="same")
        relative_blur = img_t / (conv_est + 1e-6)

        # back‑projection
        back_proj = F.conv2d(relative_blur, kernel_flip, padding="same")
        estimate = estimate * back_proj
        # optional: stabilise
        estimate = torch.clamp(estimate, min=1e-6, max=1.0)

    # ------------------------------------------------------------------
    # 3. move back to CPU numpy
    # ------------------------------------------------------------------
    result = estimate.squeeze().detach().cpu().numpy()            # (H,W)

    if to_uint8:
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
    else:
        result = result.astype(np.float32)                        # 0‑1 float

    return result
