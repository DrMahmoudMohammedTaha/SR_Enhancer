import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from torch import nn

# Define SRCNN architecture
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.srcnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.srcnn(x)

# Load models
main_path = "D:\\_research\\pedestrian_experiments\\experiment - super-resolution\\SR_Enhancer\\models\\"
yolo_model = YOLO(main_path + "yolo_model.pt")

# Initialize and load SRCNN models
srcnn_drone = SRCNN()
srcnn_drone.load_state_dict(torch.load(main_path + "srcnn_model_FarDrone.pth", map_location='cpu'))
srcnn_drone.eval()

srcnn_human = SRCNN()
srcnn_human.load_state_dict(torch.load(main_path + "srcnn_model_FarHuman.pth", map_location='cpu'))
srcnn_human.eval()

# Image transform
to_tensor = ToTensor()
to_pil = ToPILImage()

# Enhance single-channel grayscale images using SRCNN
def enhance_crop(crop, model):
    gray = crop.convert("L")  # Convert to grayscale
    input_tensor = to_tensor(gray).unsqueeze(0)  # Shape: [1, 1, H, W]
    with torch.no_grad():
        output = model(input_tensor)
    output_img = output.squeeze(0).clamp(0, 1)  # [1, H, W]
    return to_pil(output_img)

# Load image
image_path = "C:\\Users\\Mahmoud_Taha\\Desktop\\New folder\\2839_jpg.rf.f22be662b708eff20ad80889b308aa3e.jpg"
original_img = Image.open(image_path).convert("RGB")
original_cv2 = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
enhanced_img = original_cv2.copy()

# Run YOLO detection
results = yolo_model(image_path)[0]
metrics_report = []

# Loop through detections
for i, box in enumerate(results.boxes):
    cls_id = int(box.cls.item())
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cropped = original_img.crop((x1, y1, x2, y2))

    if cls_id == 2:  # Drone
        enhanced_crop = enhance_crop(cropped, srcnn_drone)
        label = "drone"
    elif cls_id == 1:  # Human
        enhanced_crop = enhance_crop(cropped, srcnn_human)
        label = "human"
    else:
        continue

    # Resize enhanced crop to match original box
    enhanced_crop_resized = enhanced_crop.resize((x2 - x1, y2 - y1))
    enhanced_cv2 = cv2.cvtColor(np.array(enhanced_crop_resized), cv2.COLOR_RGB2BGR)

    # Replace in original image
    enhanced_img[y1:y2, x1:x2] = enhanced_cv2

    # Calculate metrics
    original_crop_cv2 = cv2.cvtColor(np.array(cropped.resize((x2 - x1, y2 - y1))), cv2.COLOR_RGB2BGR)
    psnr = compare_psnr(original_crop_cv2, enhanced_cv2)
    ssim = compare_ssim(original_crop_cv2, enhanced_cv2, multichannel=True)
    metrics_report.append(f"{label} #{i + 1}: PSNR={psnr:.2f}, SSIM={ssim:.4f}")

# Save enhanced image
enhanced_path = image_path.replace(".jpg", "_enhanced.jpg")
cv2.imwrite(enhanced_path, enhanced_img)

# Save metrics report
metrics_path = image_path.replace(".jpg", "_enhanced.txt")
with open(metrics_path, "w") as f:
    f.write(f"Comparison Report for {os.path.basename(image_path)}\n")
    f.write("\n".join(metrics_report))

print(f"✅ Enhanced image saved to: {enhanced_path}")
print(f"📄 Metrics saved to: {metrics_path}")
