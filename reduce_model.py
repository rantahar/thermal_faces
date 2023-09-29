import torch
import torch.nn as nn


class FaceDetector(nn.Module):
    def __init__(self, image_width, image_height, units = 16):
        super(FaceDetector, self).__init__()
        
        self.image_width = image_width
        self.image_height = image_height
        self.units = units

        self.flat_size = 2 * units * (image_width // 4) * (image_height // 4)
        
        self.conv1 = nn.Conv2d(1, units, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(units)
        self.conv2 = nn.Conv2d(units, units, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(units, 2*units, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(2*units, 2*units, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.flat_size, 16*units)
        self.fc2 = nn.Linear(16*units, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, temperatures):
        temperatures = temperatures.unsqueeze(1)

        out = self.conv1(temperatures)
        out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        
        out = out.view(-1, self.flat_size)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        if not self.training:
            out = self.sigmoid(out)
        out = out.squeeze(1)
        return out
