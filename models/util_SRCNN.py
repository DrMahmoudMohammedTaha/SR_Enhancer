
import torch.nn.functional as F
import torch.nn as nn
import torch

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