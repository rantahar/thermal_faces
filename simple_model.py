import torch
import torch.nn as nn


class FaceDetector(nn.Module):
    def __init__(self, units):
        super(FaceDetector, self).__init__()
        self.conv_down_1 = nn.Conv2d(1, units, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(units)
        self.conv_down_2 = nn.Conv2d(units, units, kernel_size=3, stride=2, padding=1)
        self.conv_down_3 = nn.Conv2d(units, units, kernel_size=3, stride=2, padding=1)
        self.conv_down_4 = nn.Conv2d(units, 2*units, kernel_size=3, stride=2, padding=1)
        self.conv_down_5 = nn.Conv2d(2*units, 2*units, kernel_size=3, stride=2, padding=1)

        self.conv_up_5 = nn.ConvTranspose2d(2*units, 2*units, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv_up_4 = nn.ConvTranspose2d(2*units, units, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_up_3 = nn.ConvTranspose2d(2*units, units, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_up_2 = nn.ConvTranspose2d(units, units, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_up_1 = nn.ConvTranspose2d(2*units, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, temperatures):
        temperatures = temperatures.unsqueeze(1)

        out = self.conv_down_1(temperatures)
        out = self.batch_norm1(out)
        down_1 = self.relu(out)
        out = self.conv_down_2(down_1)
        out = self.relu(out)
        out = self.conv_down_3(out)
        down_3 = self.relu(out)
        out = self.conv_down_4(down_3)
        out = self.relu(out)
        out = self.conv_down_5(out)
        out = self.relu(out)

        out = self.conv_up_5(out)
        out = self.relu(out)
        out = self.conv_up_4(out)
        out = self.relu(out)
        out = torch.cat([out, down_3], dim=1)
        out = self.conv_up_3(out)
        out = self.relu(out)
        out = self.conv_up_2(out)
        out = self.relu(out)
        out = torch.cat([out, down_1], dim=1)
        out = self.conv_up_1(out)
        out = self.sigmoid(out)
        
        out = out.squeeze(1)
        return out
