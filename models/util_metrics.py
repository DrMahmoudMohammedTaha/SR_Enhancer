
from PIL import Image
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tabulate import tabulate
import csv
import os
import cv2


def calculate_sharpness(img):

    img = np.array(img.convert("L"))
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = np.mean(gx**2 + gy**2)
    print(f"sharpeness metrics: {laplacian_var}, {tenengrad} " )
    return laplacian_var, tenengrad

def calculate_metrics(enhanced, target):

    # Convert PIL to numpy
    if isinstance(enhanced, Image.Image):
        enhanced = np.array(enhanced)
    if isinstance(target, Image.Image):
        target = np.array(target)

    if isinstance(enhanced, torch.Tensor):
        enhanced = enhanced.permute(0, 2, 3, 1).cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.permute(0, 2, 3, 1).cpu().numpy()
    
    # Print shapes
    # print(f"Enhanced image shape: {enhanced.shape}")
    # print(f"Target image shape: {target.shape}")

    # If no batch dimension, add one
    if enhanced.ndim == 3:
        enhanced = np.expand_dims(enhanced, 0)
        target = np.expand_dims(target, 0)

    ssim_val = np.mean([
        ssim(o, t, data_range=t.max() - t.min(), channel_axis=-1, win_size=5)
        for o, t in zip(enhanced, target)
    ])
    psnr_val = np.mean([
        psnr(t, o, data_range=t.max() - t.min())
        for o, t in zip(enhanced, target)
    ])
    mse_val = np.mean((enhanced - target) ** 2)
    return ssim_val, psnr_val, mse_val


def export_metrics(folder_path,metrics_results):

    # Print the table
    headers = ["Image", "SSIM", "PSNR", "MSE"]
    print("\n\nComparison Table:")
    print(tabulate(metrics_results, headers=headers, tablefmt="grid"))

    # Save to CSV
    csv_path = os.path.join(folder_path, "_metrics_comparison.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "SSIM", "PSNR", "MSE"])  # Header
        writer.writerows(metrics_results)
        
        
    # Compute and print averages (only valid values)
    valid_metrics = [row for row in metrics_results if "ERROR" not in row]
    if valid_metrics:
        avg_ssim = np.mean([float(row[1]) for row in valid_metrics])
        avg_psnr = np.mean([float(row[2]) for row in valid_metrics])
        avg_mse = np.mean([float(row[3]) for row in valid_metrics])

        summary_path = os.path.join(folder_path, "_average_metrics.txt")
        with open(summary_path, "w") as f:
            f.write("Average Image Quality Metrics\n")
            f.write("=============================\n")
            f.write(f"Average SSIM: {avg_ssim:.4f}\n")
            f.write(f"Average PSNR: {avg_psnr:.2f}\n")
            f.write(f"Average MSE: {avg_mse:.4f}\n")

        print(f"\nAverage SSIM: {avg_ssim:.4f}")
        print(f"Average PSNR: {avg_psnr:.2f}")
        print(f"Average MSE: {avg_mse:.4f}")
    else:
        print("\nNo valid images processed for average metrics.")
