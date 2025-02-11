import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.models import vgg16
from collections import namedtuple
from utilities import toTensor255, toPil


# From https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/vgg.py
class Vgg16(torch.nn.Module):
    def __init__(self, device="cpu"):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = vgg16(weights="VGG16_Weights.IMAGENET1K_V1").features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x].to(device))
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x].to(device))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.pad = torch.nn.ReflectionPad2d(int(np.floor(kernel_size / 2)))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.leakyrelu(x)
        return x


class Join(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.norm1 = nn.BatchNorm2d(c1)
        self.norm2 = nn.BatchNorm2d(c2)

    def forward(self, x, y):
        x = self.upsample(x)
        x = self.norm1(x)
        y = self.norm2(y)
        return torch.cat([x, y], 1)


class TextureNetworks(nn.Module):
    def __init__(self):
        super().__init__()
        # 16x16
        self.seq16 = nn.Sequential(
            ConvBlock(3, 8, 3),
            ConvBlock(8, 8, 3),
            ConvBlock(8, 8, 1),
        )

        # 32x32
        self.seq32_1 = nn.Sequential(
            ConvBlock(3, 8, 3),
            ConvBlock(8, 8, 3),
            ConvBlock(8, 8, 1),
        )
        self.join16_32 = Join(8, 8)
        self.seq32_2 = nn.Sequential(
            ConvBlock(16, 16, 3),
            ConvBlock(16, 16, 3),
            ConvBlock(16, 16, 1),
        )

        # 64x64
        self.seq64_1 = nn.Sequential(
            ConvBlock(3, 8, 3),
            ConvBlock(8, 8, 3),
            ConvBlock(8, 8, 1),
        )
        self.join32_64 = Join(16, 8)
        self.seq64_2 = nn.Sequential(
            ConvBlock(24, 24, 3),
            ConvBlock(24, 24, 3),
            ConvBlock(24, 24, 1),
        )

        # 128x128
        self.seq128_1 = nn.Sequential(
            ConvBlock(3, 8, 3),
            ConvBlock(8, 8, 3),
            ConvBlock(8, 8, 1),
        )
        self.join64_128 = Join(24, 8)
        self.seq128_2 = nn.Sequential(
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 3),
            ConvBlock(32, 32, 1),
        )

        # 256x256
        self.seq256_1 = nn.Sequential(
            ConvBlock(3, 8, 3),
            ConvBlock(8, 8, 3),
            ConvBlock(8, 8, 1),
        )
        self.join128_256 = Join(32, 8)
        self.seq256_2 = nn.Sequential(
            ConvBlock(40, 40, 3),
            ConvBlock(40, 40, 3),
            ConvBlock(40, 40, 1),
        )

        # channel reduction
        self.conv = nn.Conv2d(40, 3, 1)

    def forward(self, x256):
        x16 = F.interpolate(x256, scale_factor=(1 / 16), mode="bilinear")
        x32 = F.interpolate(x256, scale_factor=(1 / 8), mode="bilinear")
        x64 = F.interpolate(x256, scale_factor=(1 / 4), mode="bilinear")
        x128 = F.interpolate(x256, scale_factor=(1 / 2), mode="bilinear")

        x16 = self.seq16(x16)

        x32 = self.seq32_1(x32)
        x32 = self.join16_32(x16, x32)
        x32 = self.seq32_2(x32)

        x64 = self.seq64_1(x64)
        x64 = self.join32_64(x32, x64)
        x64 = self.seq64_2(x64)

        x128 = self.seq128_1(x128)
        x128 = self.join64_128(x64, x128)
        x128 = self.seq128_2(x128)

        x256 = self.seq256_1(x256)
        x256 = self.join128_256(x128, x256)
        x256 = self.seq256_2(x256)

        x256 = self.conv(x256)
        return x256


if __name__ == "__main__":
    model = TextureNetworks()
    model.load_state_dict(torch.load("./models/Coco2014_epoch_2_batchSize_4.pth", weights_only=True), strict=True)
    img = Image.open("./123.jpg").convert("RGB")
    img = toTensor255(img).unsqueeze(0)
    img = model(img).squeeze(0)
    img = toPil(img.byte())
    img.save("./output.jpg")
