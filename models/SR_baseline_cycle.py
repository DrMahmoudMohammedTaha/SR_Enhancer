
import os
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageFilter
from pathlib import Path

from util_metrics import calculate_metrics, export_metrics, calculate_sharpness
from util_SRCNN import SRCNN

main_path = "D:\\_research\\pedestrian_experiments\\experiment - super-resolution\\SR_Enhancer\\models\\"

srcnn_baseline = SRCNN.load_srcnn_model(main_path + 'srcnn_model.pth')
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

    upsampled_img = upsampled_img.filter(ImageFilter.SHARPEN)
    enhanced_img = enhance_crop(upsampled_img, srcnn_baseline)

    
    ssim_val, psnr_val, mse_val = calculate_metrics(enhanced_img, original_img)
    calculate_sharpness(original_img)
    calculate_sharpness(upsampled_img)
    calculate_sharpness(enhanced_img)
    # Save metrics to txt
    metrics_path = image_path.replace(".jpg", "_enhanced.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Comparison Report for {os.path.basename(image_path)}\n")
        f.write(f"PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f}, MES={mse_val:.4f}")

    # Save enhanced image
    enhanced_path = image_path.replace(".jpg", "_enhanced.jpg")
    enhanced_img.save(enhanced_path)
    # cv2.imwrite(enhanced_path, np.array(enhanced_img))
    
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

folder_path = "C:\\Users\\Mahmoud_Taha\\Desktop\\SR_baseline_images"
apply_folder_MoE(folder_path)