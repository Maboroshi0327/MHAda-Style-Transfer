import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class ConvDepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        pad = int(np.floor(kernel_size / 2))
        self.pad = torch.nn.ReflectionPad2d(pad)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.pad(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        pad = int(np.floor(kernel_size / 2))
        self.pad = torch.nn.ReflectionPad2d(pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        return x


class ConvReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvTanh(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.tanh(x)
        x = (x + 1) / 2 * 255
        return x


class ConvReluInterpolate(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, scale_factor: float):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU()
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvReluInterpolate(512, 256, 3, 1, 2),
            ConvReLU(256, 256, 3, 1),
            ConvReLU(256, 256, 3, 1),
            ConvReLU(256, 256, 3, 1),
            ConvReluInterpolate(256, 128, 3, 1, 2),
        )

        self.conv2 = nn.Sequential(
            ConvReLU(128, 128, 3, 1),
            ConvReluInterpolate(128, 64, 3, 1, 2),
        )

        self.conv3 = nn.Sequential(
            ConvReLU(64, 64, 3, 1),
            ConvReLU(64, 3, 3, 1),
        )

    def forward(self, fcs: torch.Tensor):
        fcs = self.conv1(fcs)
        fcs = self.conv2(fcs)
        cs = self.conv3(fcs)
        return cs
