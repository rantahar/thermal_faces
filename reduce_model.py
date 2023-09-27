import torch
import torch.nn as nn


class FaceDetectionModel(nn.Module):
    def __init__(self, image_width, image_height, units = 16):
        super(FaceDetectionModel, self).__init__()
        
        self.image_width = image_width
        self.image_height = image_height
        
        self.conv1 = nn.Conv2d(1, units, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(units)
        self.conv2 = nn.Conv2d(units, 2*units, kernel_size=3, stride=2, padding=1)
        self.fc1 = nn.Linear(2 * units * (image_width // 4) * (image_height // 4), 16*units)
        self.fc2 = nn.Linear(16*units, 1)

        self.relu = nn.ReLU()

    def forward(self, input):
        out = self.conv1(input)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        
        out = out.view(-1, 32 * (self.image_width // 4) * (self.image_height // 4))
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
