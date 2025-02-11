from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearProjection(nn.Module):
    def __init__(self, img_size: tuple = (360, 640), patch_size: tuple = (8, 8), in_chans=3, embed_dim=512):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x


class CAPE(nn.Module):
    def __init__(self, in_chans: int = 512, out_chans: int = 512):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(18)
        self.pos = nn.Conv2d(in_chans, out_chans, (1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.avgpool(x)
        x = self.pos(x)
        x = F.interpolate(x, mode="bilinear", size=(H, W))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.MultiheadAttention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout2_1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2_2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.relu = nn.ReLU()

    def positional_encoding(self, tensor: torch.Tensor, pos: Optional[torch.Tensor]):
        if pos is None:
            return tensor
        return tensor + pos

    def forward(self, x: torch.Tensor, pos: Optional[torch.Tensor] = None):
        x = self.positional_encoding(x, pos)
        q = k = v = x = x.flatten(2).permute(0, 2, 1)

        y, _ = self.MultiheadAttention(q, k, v)
        y = self.dropout1(y)
        x = x + y
        x = self.norm1(x)

        y = self.linear1(x)
        y = self.relu(y)
        y = self.dropout2_1(y)
        y = self.linear2(y)
        y = self.dropout2_2(y)
        x = x + y
        x = self.norm2(x)
        return x


if __name__ == "__main__":
    img = torch.randn(1, 3, 360, 640)
    proj = LinearProjection()
    x1 = proj(img)
    print(x1.shape)

    cape = CAPE()
    x2 = cape(x1)
    print(x2.shape)

    enc = TransformerEncoder()
    x3 = enc(x1, x2)
    print(x3.shape)
