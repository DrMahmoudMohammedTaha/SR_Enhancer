import os
import torch
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
from ultralytics import YOLO
from PIL import Image, ImageFilter
from pathlib import Path


from util_metrics import calculate_metrics, export_metrics, calculate_sharpness
from util_SRCNN import SRCNN

main_path = "D:\\_research\\pedestrian_experiments\\experiment - super-resolution\\SR_Enhancer\\models\\"
yolo_model = YOLO(main_path + "best.pt")
srcnn_drone = SRCNN.load_srcnn_model(main_path + 'srcnn_model_FarDrone.pth')
srcnn_human = SRCNN.load_srcnn_model(main_path + 'srcnn_model_FarHuman.pth')
downsampling_ratio = 2

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

def apply_image_MoE(image_path):

    # original_img = Image.open(image_path).convert("RGB")
    # original_cv2 = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    # enhanced_img = original_cv2.copy()

    # Load original image
    original_img = Image.open(image_path).convert("RGB")

    # Step 1: Downsample the original image to simulate low-resolution
    low_res_img = original_img.resize(
        (original_img.width // downsampling_ratio, original_img.height // downsampling_ratio), Image.BICUBIC
    )

    # Step 2: Upsample it back to original resolution
    upsampled_img = low_res_img.resize(
        (original_img.width, original_img.height), Image.BICUBIC
    )
    # upsampled_np = cv2.cvtColor(np.array(upsampled_img), cv2.COLOR_RGB2BGR)
    # # upsampled_np = np.array(upsampled_img)
    # enhanced_img = upsampled_np.copy()

    upsampled_img = upsampled_img.filter(ImageFilter.SHARPEN)
    enhanced_img = np.array(upsampled_img)
    # Detect objects
    results = yolo_model(image_path)[0]

    print("Detections: " + str(len(results.boxes)))
    for i, box in enumerate(results.boxes):

        cls_id = int(box.cls.item())
        print("cls_id: " + str(cls_id))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        enhanced_crop = upsampled_img.crop((x1, y1, x2, y2))

        if cls_id == 2:  # Drone
            enhanced_crop = enhance_crop(enhanced_crop, srcnn_drone)
        elif cls_id == 1:  # Human
            enhanced_crop = enhance_crop(enhanced_crop, srcnn_human)
        else:
            continue  # Skip unknown class

        # Resize enhanced crop to original size
        enhanced_crop_resized = enhanced_crop.resize((x2 - x1, y2 - y1))
        enhanced_img[y1:y2, x1:x2] = enhanced_crop_resized

    ssim_val, psnr_val, mse_val = calculate_metrics(enhanced_img, original_img)


    # Save metrics to txt
    metrics_path = image_path.replace(".jpg", "_enhanced.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Comparison Report for {os.path.basename(image_path)}\n")
        f.write(f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MES={mse_val:.4f}")

    # Save enhanced image
    enhanced_path = image_path.replace(".jpg", "_enhanced.jpg")

    enhanced_img = Image.fromarray(enhanced_img)

    calculate_sharpness(original_img)
    calculate_sharpness(upsampled_img)
    calculate_sharpness(enhanced_img)

    enhanced_img.save(enhanced_path)
    
    print(f"Enhanced Image and Report saved: {enhanced_path}")

    return os.path.basename(image_path), ssim_val, psnr_val, mse_val

def apply_folder_MoE(folder_path):
    folder = Path(folder_path)
    image_paths = list(folder.rglob("*.jpg"))

    print(f"Found {len(image_paths)} images in {folder_path}")
    metrics_results = []

    for image_path in image_paths:

        if('enhanced' in str(image_path)):
            continue
        print(f"\nProcessing: {image_path}")
        try:
            filename, ssim_val, psnr_val, mse_val = apply_image_MoE(str(image_path))
            metrics_results.append([filename, f"{ssim_val:.4f}", f"{psnr_val:.2f}", f"{mse_val:.4f}"])
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")
            metrics_results.append([image_path.name, "ERROR", "ERROR", "ERROR"])

    export_metrics(folder_path,metrics_results)

# image_path = "C:\\Users\\Mahmoud_Taha\\Desktop\\New folder\\2839_jpg.rf.f22be662b708eff20ad80889b308aa3e.jpg"
# apply_image_MoE(image_path)

folder_path = "C:\\Users\\Mahmoud_Taha\\Desktop\\SR_MoE_images"
apply_folder_MoE(folder_path)