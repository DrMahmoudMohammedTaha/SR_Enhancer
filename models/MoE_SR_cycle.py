import os
import cv2
import torch
import csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
from tabulate import tabulate

class SRCNN(nn.Module): # neural network module 
    def __init__(self):
        super(SRCNN, self).__init__() # calls the constructor of the parent class (nn.Module) # it ensures that the nn.Module part of the SRCNN object is properly set up.
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    @staticmethod
    def load_srcnn_model(path):
        model = SRCNN()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()
        return model

main_path = "D:\\_research\\pedestrian_experiments\\experiment - super-resolution\\SR_Enhancer\\models\\"
yolo_model = YOLO(main_path + "best.pt")
srcnn_drone = SRCNN.load_srcnn_model(main_path + 'srcnn_model_FarDrone.pth')
srcnn_human = SRCNN.load_srcnn_model(main_path + 'srcnn_model_FarHuman.pth')

# Image transform
to_tensor = ToTensor()
to_pil = ToPILImage()

# Define function to enhance cropped object
def enhance_crop(crop, model):
    with torch.no_grad():
        input_tensor = to_tensor(crop).unsqueeze(0)
        enhanced = model(input_tensor)
        output_img = enhanced.squeeze(0).clamp(0, 1)
        return to_pil(output_img)

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
    print(f"Enhanced image shape: {enhanced.shape}")
    print(f"Target image shape: {target.shape}")

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


def apply_image_MoE(image_path):

    # original_img = Image.open(image_path).convert("RGB")
    # original_cv2 = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    # enhanced_img = original_cv2.copy()

    # Load original image
    original_img = Image.open(image_path).convert("RGB")
    original_np = np.array(original_img)  # For metric comparison later

    # Step 1: Downsample the original image to simulate low-resolution
    low_res_img = original_img.resize(
        (original_img.width // 2, original_img.height // 2), Image.BICUBIC
    )

    # Step 2: Upsample it back to original resolution
    upsampled_img = low_res_img.resize(
        (original_img.width, original_img.height), Image.BICUBIC
    )
    upsampled_np = cv2.cvtColor(np.array(upsampled_img), cv2.COLOR_RGB2BGR)
    # upsampled_np = np.array(upsampled_img)
    enhanced_img = upsampled_np.copy()

    # Detect objects
    results = yolo_model(image_path)[0]

    print("Detections: " + str(len(results.boxes)))
    for i, box in enumerate(results.boxes):

        cls_id = int(box.cls.item())
        print("cls_id: " + str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cropped = upsampled_img.crop((x1, y1, x2, y2))

        if cls_id == 20:  # Drone
            enhanced_crop = enhance_crop(cropped, srcnn_drone)
            label = "drone"
        elif cls_id == 10:  # Human
            enhanced_crop = enhance_crop(cropped, srcnn_human)
            label = "human"
        else:
            continue  # Skip unknown class

        # Resize enhanced crop to original size
        enhanced_crop_resized = enhanced_crop.resize((x2 - x1, y2 - y1))
        enhanced_cv2 = cv2.cvtColor(np.array(enhanced_crop_resized), cv2.COLOR_RGB2BGR)

        # Replace original crop with enhanced version
        enhanced_img[y1:y2, x1:x2] = enhanced_cv2

    ssim_val, psnr_val, mse_val = calculate_metrics(enhanced_img, original_img)

    # Save metrics to txt
    metrics_path = image_path.replace(".jpg", "_enhanced.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Comparison Report for {os.path.basename(image_path)}\n")
        f.write(f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MES={mse_val:.4f}")

    # Save enhanced image
    enhanced_path = image_path.replace(".jpg", "_enhanced.jpg")
    cv2.imwrite(enhanced_path, enhanced_img)

    print(f"Enhanced Image and Report saved: {enhanced_path}")

    return os.path.basename(image_path), ssim_val, psnr_val, mse_val

def apply_folder_MoE(folder_path):
    folder = Path(folder_path)
    image_paths = list(folder.rglob("*.jpg"))

    print(f"Found {len(image_paths)} images in {folder_path}")
    metrics_results = []

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        try:
            filename, ssim_val, psnr_val, mse_val = apply_image_MoE(str(image_path))
            metrics_results.append([filename, f"{ssim_val:.4f}", f"{psnr_val:.2f}", f"{mse_val:.4f}"])
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            metrics_results.append([image_path.name, "ERROR", "ERROR", "ERROR"])

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



# image_path = "C:\\Users\\Mahmoud_Taha\\Desktop\\New folder\\2839_jpg.rf.f22be662b708eff20ad80889b308aa3e.jpg"
# apply_image_MoE(image_path)

folder_path = "C:\\Users\\Mahmoud_Taha\\Desktop\\New folder"
apply_folder_MoE(folder_path)