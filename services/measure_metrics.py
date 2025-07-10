import cv2
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def compute_psnr_torch(img1, img2):
#     mse = F.mse_loss(img1, img2)
#     if mse == 0:
#         return float("inf")
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_psnr_torch(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return torch.tensor(float('inf')).to(img1.device)
    return 20 * torch.log10(torch.tensor(1.0, device=img1.device) / torch.sqrt(mse))

def compute_ssim_torch(img1, img2, window_size=11, size_average=True):
    # Gaussian window
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size//2)**2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window @ _1D_window.T
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window.to(device)

    (_, channel, height, width) = img1.size()
    window = create_window(window_size, channel)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean([1, 2, 3])

def preprocess_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return rgb

def compute_psnr_ssim_torch(original_frame, enhanced_frame):
    original_tensor = preprocess_frame(original_frame)
    enhanced_tensor = preprocess_frame(enhanced_frame)

    psnr_val = compute_psnr_torch(original_tensor, enhanced_tensor).item()
    ssim_val = compute_ssim_torch(original_tensor, enhanced_tensor).item()

    return psnr_val, ssim_val
