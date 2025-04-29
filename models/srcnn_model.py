import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = "models\\srcnn_model.pth"
model = None

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

def load_model(model_path: str):

    global model
    global device
    global transform

    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model


def apply_ai_model(frame):

    global model
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