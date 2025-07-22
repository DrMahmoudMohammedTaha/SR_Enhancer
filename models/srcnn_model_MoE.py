import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from torch import nn
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

yolo_MOE = None
srcnn_drone = None
srcnn_human = None
to_tensor  = None
to_pil = None

# Define SRCNN architecture
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

def load_MOE_model(model_path: str):

    global device
    global transform
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def load_MOE():
    
    try:
        global yolo_MOE
        global srcnn_drone
        global srcnn_human
        global to_tensor
        global to_pil

        yolo_MOE = YOLO("models\\yolo_model.pt")

        # Initialize and load SRCNN models
        srcnn_drone = load_MOE_model("models\\srcnn_model_FarDrone.pth")
        srcnn_human = load_MOE_model("models\\srcnn_model_FarHuman.pth")

        # Image transform
        to_tensor = ToTensor()
        to_pil = ToPILImage()

        return True
    except:
        return False

def apply_SR_model_(frame , model):

    global device
    global transform
            
    try:
        # Convert frame to PIL Image
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Convert to PIL Image and apply transformations
        pil_img = Image.fromarray(frame)
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Process with model
        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255).astype(np.uint8)
        
        return output
    except Exception as e:
        print(f"Error applying AI model: {str(e)}")
        return frame

# def apply_MOE_model(frame):
#     # Run YOLO detection
#     results = yolo_MOE(frame)[0]

#     # Loop through detections
#     for i, box in enumerate(results.boxes):
#         cls_id = int(box.cls.item())
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         cropped = frame.crop((x1, y1, x2, y2))

#         if cls_id == 2:  # Drone
#             enhanced_crop = apply_SR_model_(cropped, srcnn_drone)
#         elif cls_id == 1:  # Human
#             enhanced_crop = apply_SR_model_(cropped, srcnn_human)
#         else:
#             continue

#         # Resize enhanced crop to match original box
#         enhanced_crop_resized = enhanced_crop.resize((x2 - x1, y2 - y1))
#         enhanced_cv2 = cv2.cvtColor(np.array(enhanced_crop_resized), cv2.COLOR_RGB2BGR)

#         # Replace in original image
#         frame[y1:y2, x1:x2] = enhanced_cv2

#         return frame


def apply_MOE_model(frame):
    # Run YOLO detection
    results = yolo_MOE(frame)[0]

    # Convert NumPy frame to PIL Image (needed for crop)
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Loop through detections
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        cropped = pil_frame.crop((x1, y1, x2, y2))

        if cls_id == 2:  # Drone
            enhanced_crop = apply_SR_model_(np.array(cropped), srcnn_drone)
        elif cls_id == 1:  # Human
            enhanced_crop = apply_SR_model_(np.array(cropped), srcnn_human)
        else:
            continue

        # Resize enhanced crop to match original box
        enhanced_crop_resized = Image.fromarray(enhanced_crop).resize((x2 - x1, y2 - y1))
        enhanced_cv2 = cv2.cvtColor(np.array(enhanced_crop_resized), cv2.COLOR_RGB2BGR)

        # Replace in original image
        frame[y1:y2, x1:x2] = enhanced_cv2

    return frame