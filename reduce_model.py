import torch
import torch.nn as nn

class FaceDetectionModel(nn.Module):
    def __init__(self, image_width, image_height):
        super(FaceDetectionModel, self).__init__()
        
        self.image_width = image_width
        self.image_height = image_height
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (image_width // 4) * (image_height // 4), 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 32 * (self.image_width // 4) * (self.image_height // 4))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
        