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
        self.conv2 = nn.Conv2d(units, units, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(units, 2*units, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(2*units, 2*units, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self.flat_size, 16*units)
        self.fc2 = nn.Linear(16*units, 1)

        #self.batch_norm1 = nn.BatchNorm2d(units)
        #self.batch_norm3 = nn.BatchNorm2d(2*units)

        self.relu = nn.ReLU()

    def forward(self, temperatures):
        assert(temperatures.shape[1] == self.image_height)
        assert(temperatures.shape[2] == self.image_width)
        temperatures = temperatures.unsqueeze(1)
        out = temperatures/40

        out = self.conv1(out)
        #out = self.batch_norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        #out = self.batch_norm3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.relu(out)
        
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        out = out.squeeze(1)
        if not self.training:
            out = nn.functional.sigmoid(out)
        return out
