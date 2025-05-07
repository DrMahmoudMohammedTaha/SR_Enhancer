import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


SR_model_loaded = False
transform = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SR_model_path = "models\\srcnn_model.pth"
SR_model = None

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

def load_SR_model(SR_model_path: str):

    global SR_model
    global device
    global transform
    global SR_model_loaded


    try:
        SR_model = SRCNN().to(device)
        SR_model.load_state_dict(torch.load(SR_model_path, map_location=device))
        SR_model.eval()
        SR_model_loaded = True

    except Exception as e:
        print(f"Error loading SR_model: {str(e)}")
        SR_model_loaded = False

    return SR_model_loaded


def apply_SR_model(frame):

    global SR_model
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

        # Process with SR_model
        with torch.no_grad():
            output = SR_model(input_tensor)
            output = output.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))
            output = (output * 255).astype(np.uint8)
        
        return output
    except Exception as e:
        print(f"Error applying SR SR_model: {str(e)}")
        return frame