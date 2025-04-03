import torch
import torch.nn as nn
from torch import linalg as LA
from torch.nn import functional as F

import numpy as np

from vit import ViT_MultiScale, ViT_torch


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
    def __init__(self, multi_scale: bool = True):
        super().__init__()
        self.conv1 = nn.Sequential(
            ConvReLU(512, 512, 3, 1),
            ConvReLU(512, 512, 3, 1),
            ConvReluInterpolate(512, 256, 3, 1, 2) if multi_scale else ConvReLU(512, 256, 3, 1),
        )

        self.ada2conv = ConvReLU(512, 256, 3, 1)
        self.conv2 = nn.Sequential(
            ConvReLU(256, 256, 3, 1),
            ConvReLU(256, 256, 3, 1),
            ConvReluInterpolate(256, 128, 3, 1, 2),
        )

        self.ada1conv = ConvReLU(512, 128, 3, 1) if multi_scale else ConvReluInterpolate(512, 128, 3, 1, 2)
        self.conv3 = nn.Sequential(
            ConvReLU(128, 128, 3, 1),
            ConvReLU(128, 128, 3, 1),
            ConvReluInterpolate(128, 64, 3, 1, 2),
        )

        self.conv4 = nn.Sequential(
            ConvReLU(64, 64, 3, 1),
            ConvReluInterpolate(64, 32, 3, 1, 2),
        )

        self.conv5 = nn.Sequential(
            ConvReLU(32, 32, 3, 1),
            ConvReLU(32, 3, 3, 1),
        )

    def forward(self, ada3, ada2, ada1):
        x = self.conv1(ada3)

        ada2 = self.ada2conv(ada2)
        x = x + ada2
        x = self.conv2(x)

        ada1 = self.ada1conv(ada1)
        x = x + ada1
        x = self.conv3(x)

        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k):
        return self.softmax(torch.bmm(q, k))


class CosineSimilarity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k):
        """
        q:   (b, t, d)
        k:   (b, d, t)
        out: (b, t, t)
        """
        q_norm = LA.vector_norm(q, dim=-1, keepdim=True)
        k_norm = LA.vector_norm(k, dim=1, keepdim=True)
        s = torch.bmm(q, k) / torch.bmm(q_norm, k_norm) + 1
        a = s / s.sum(dim=-1, keepdim=True)
        return a


class AdaAttN(nn.Module):
    def __init__(self, height, width, qkv_dim, activation="softmax"):
        super().__init__()
        self.f = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.g = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.h = nn.Conv2d(qkv_dim, qkv_dim, 1)
        self.norm_q = nn.InstanceNorm2d(qkv_dim, affine=False)
        self.norm_k = nn.InstanceNorm2d(qkv_dim, affine=False)
        self.norm_v = nn.InstanceNorm2d(qkv_dim, affine=False)

        self.qkv_dim = qkv_dim
        self.height = height
        self.width = width

        if activation == "softmax":
            self.activation = Softmax()
        elif activation == "cosine":
            self.activation = CosineSimilarity()
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, fc: torch.Tensor, fs: torch.Tensor):
        fc = fc.permute(0, 2, 1).reshape(-1, self.qkv_dim, self.height, self.width)
        fs = fs.permute(0, 2, 1).reshape(-1, self.qkv_dim, self.height, self.width)

        # Q^T
        Q = self.f(self.norm_q(fc))
        b, _, h, w = Q.size()
        Q = Q.view(b, -1, h * w).permute(0, 2, 1)

        # K
        K = self.g(self.norm_k(fs))
        b, _, h, w = K.size()
        K = K.view(b, -1, h * w)

        # V^T
        V = self.h(fs)
        b, _, h, w = V.size()
        V = V.view(b, -1, h * w).permute(0, 2, 1)

        # A * V^T
        A = self.activation(Q, K)
        M = torch.bmm(A, V)

        # S
        Var = torch.bmm(A, V**2) - M**2
        S = torch.sqrt(Var.clamp(min=1e-6))

        # Reshape M and S
        b, _, h, w = fc.size()
        M = M.view(b, h, w, -1).permute(0, 3, 1, 2)
        S = S.view(b, h, w, -1).permute(0, 3, 1, 2)

        return S * self.norm_v(fc) + M


class AdaViT(nn.Module):
    def __init__(self, activation="softmax"):
        super().__init__()
        self.adaattn1 = AdaAttN(32, 32, qkv_dim=512, activation=activation)
        self.adaattn2 = AdaAttN(32, 32, qkv_dim=512, activation=activation)
        self.adaattn3 = AdaAttN(32, 32, qkv_dim=512, activation=activation)

        self.decoder = Decoder(multi_scale=False)

    def forward(self, fc, fs):
        ada1 = self.adaattn1(fc[0], fs[0])
        ada2 = self.adaattn2(fc[1], fs[1])
        ada3 = self.adaattn3(fc[2], fs[2])
        cs = self.decoder(ada3, ada2, ada1)
        return cs


class AdaMSViT(nn.Module):
    def __init__(self, image_size: tuple, patch_size: int = 4, activation="softmax"):
        super().__init__()
        patch_h = image_size[0] // patch_size
        patch_w = image_size[1] // patch_size

        self.adaattn1 = AdaAttN(patch_h, patch_w, qkv_dim=512, activation=activation)
        self.adaattn2 = AdaAttN(patch_h // 2, patch_w // 2, qkv_dim=512, activation=activation)
        self.adaattn3 = AdaAttN(patch_h // 4, patch_w // 4, qkv_dim=512, activation=activation)

        self.decoder = Decoder(multi_scale=True)

    def forward(self, fc, fs):
        ada1 = self.adaattn1(fc[0], fs[0])
        ada2 = self.adaattn2(fc[1], fs[1])
        ada3 = self.adaattn3(fc[2], fs[2])
        cs = self.decoder(ada3, ada2, ada1)
        return cs


def test_AdaViT():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a random tensor and normalize it
    c = torch.rand(4, 3, 256, 256).to(device)
    s = torch.rand(4, 3, 256, 256).to(device)

    # Create a StylizingNetwork model and forward propagate the input tensor
    vit_c = ViT_torch(pos_embedding=True).to(device)
    vit_s = ViT_torch(pos_embedding=False).to(device)
    model = AdaViT().to(device)

    # Forward pass
    fc = vit_c(c)
    fs = vit_s(s)
    cs = model(fc, fs)
    print(c.shape)
    print(cs.shape)


def test_AdaMSViT():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_size = (256, 512)
    activation = "softmax"

    # Load a random tensor and normalize it
    c = torch.rand(8, 3, image_size[0], image_size[1]).to(device)
    s = torch.rand(8, 3, image_size[0], image_size[1]).to(device)

    # Create a StylizingNetwork model and forward propagate the input tensor
    vit_c = ViT_MultiScale(pos_embedding=True).to(device)
    vit_s = ViT_MultiScale(pos_embedding=False).to(device)
    model = AdaMSViT(image_size, activation=activation).to(device)

    # Forward pass
    fc = vit_c(c)
    fs = vit_s(s)
    cs = model(fc, fs)
    print(c.shape)
    print(cs.shape)


if __name__ == "__main__":
    test_AdaMSViT()
