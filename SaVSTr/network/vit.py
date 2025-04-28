import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import List

from .conv import ConvDepthwiseSeparable


class ConvFF(nn.Module):
    def __init__(self, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.convIN = nn.Conv2d(hidden_dim, mlp_dim, kernel_size=1)
        self.convDW = ConvDepthwiseSeparable(mlp_dim, mlp_dim, kernel_size=3, stride=1)
        self.convOUT = nn.Conv2d(mlp_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor, h: int, w: int):
        b, _, c = x.shape
        x = x.permute(0, 2, 1).reshape(b, c, h, w)
        x = self.convIN(x)
        x = self.convDW(x)
        x = self.convOUT(x)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        return x


class EncoderBlockConvFF(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.mlp = ConvFF(hidden_dim=hidden_dim, mlp_dim=mlp_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, input: torch.Tensor, h: int, w: int):
        x = self.ln1(input)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = x + input

        y = self.ln2(x)
        y = self.mlp(y, h, w)
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, num_heads: int, hidden_dim: int, mlp_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, hidden_dim),
        )
        self.ln1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, input: torch.Tensor):
        x = self.ln1(input)
        x, _ = self.attention(x, x, x, need_weights=False)
        x = x + input

        y = self.ln2(x)
        y = self.mlp(y)
        return x + y


class PosEmbedding(nn.Module):
    def __init__(self, patch_size: int = 8, embed_dim: int = 512, base_grid_size: int = 32):
        """
        embed_dim: dimension of each token
        base_grid_size: resolution used to generate the initial coordinate grid
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.base_grid_size = base_grid_size

        # Use a CNN to map 2D coordinates to the embed_dim dimension
        self.conv = nn.Conv2d(2, embed_dim, kernel_size=3, padding=1)

        # Pre-construct a fixed resolution coordinate grid with values ranging from -1 to 1
        grid_x = torch.linspace(-1, 1, self.base_grid_size)
        grid_y = torch.linspace(-1, 1, self.base_grid_size)
        grid = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid = torch.stack(grid, dim=0).unsqueeze(0)  # shape: (1, 2, base_grid_size, base_grid_size)
        self.register_buffer("grid", grid)

    def forward(self, x_shape: torch.Size) -> torch.Tensor:
        """
        x_shape: shape of the input tensor (B, C, H, W)
        Returns the CNN-based positional embedding, adjusted to have the same shape as x (B, N, embed_dim)
        """
        b, _, h, w = x_shape
        out_h = h // self.patch_size
        out_w = w // self.patch_size

        # Map 2D coordinates to embed_dim dimension using the CNN (grid is precomputed in __init__)
        pos_embed = self.conv(self.grid)

        # Use bilinear interpolation to match the spatial dimensions of the input
        pos_embed = F.interpolate(pos_embed, size=(out_h, out_w), mode="bilinear", align_corners=False)

        # Expand to match the batch size
        pos_embed = pos_embed.expand(b, -1, -1, -1)

        # Reshape and permute to get shape: (B, N, embed_dim)
        pos_embed = pos_embed.reshape(b, self.embed_dim, out_h * out_w).permute(0, 2, 1)

        return pos_embed


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, hidden_dim: int):
        super().__init__()
        # Convolutional layer to map image patches to a higher dimensional space (hidden_dim).
        self.conv_proj = nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply convolution to partition the image into patches.
        x = self.conv_proj(x)
        b, c, h, w = x.shape
        # Reshape and permute tensor to get shape: (batch, num_patches, hidden_dim)
        x = x.reshape(b, c, h * w).permute(0, 2, 1)
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 8,
        num_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 512,
        mlp_dim: int = 2048,
        pos_embedding: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, hidden_dim=hidden_dim)
        self.pos_embedding = PosEmbedding(patch_size=patch_size, embed_dim=hidden_dim) if pos_embedding else None
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_shape = x.shape
        out_h = x_shape[2] // self.patch_size
        out_w = x_shape[3] // self.patch_size

        # Apply patch embedding to the input image
        x = self.patch_embedding(x)

        # Add positional embedding if enabled
        if self.pos_embedding is not None:
            pos_embed = self.pos_embedding(x_shape)
            x = x + pos_embed

        # Pass through each encoder layer
        z = list()
        for layer in self.encoder:
            x = layer(x)
            y = x.permute(0, 2, 1)
            y = y.reshape(-1, self.hidden_dim, out_h, out_w)
            z.append(y)

        return z


class VisionTransformerMultiScale(nn.Module):
    def __init__(
        self,
        patch_size: int = 4,
        num_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: List = [256, 512, 512],
        mlp_dim: List = [1024, 2048, 2048],
        pos_embedding: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.patch_embedding = PatchEmbedding(in_channels=3, patch_size=patch_size, hidden_dim=hidden_dim[0])
        self.pos_embedding = PosEmbedding(patch_size=patch_size, embed_dim=hidden_dim[0]) if pos_embedding else None
        self.encoder = nn.ModuleList(
            [
                EncoderBlock(
                    num_heads=num_heads,
                    hidden_dim=hidden_dim[i],
                    mlp_dim=mlp_dim[i],
                )
                for i in range(num_layers)
            ]
        )
        self.repatch = nn.ModuleList(
            [
                PatchEmbedding(
                    in_channels=hidden_dim[i],
                    patch_size=2,
                    hidden_dim=hidden_dim[i + 1],
                )
                for i in range(num_layers - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x_shape = x.shape
        out_h = x_shape[2] // self.patch_size
        out_w = x_shape[3] // self.patch_size

        # Apply patch embedding to the input image
        x = self.patch_embedding(x)

        # Add positional embedding if enabled
        if self.pos_embedding is not None:
            pos_embed = self.pos_embedding(x_shape)
            x = x + pos_embed

        # Pass through each encoder layer
        z = list()
        for i in range(self.num_layers):
            x = self.encoder[i](x)
            y = x.permute(0, 2, 1)
            y = y.reshape(-1, self.hidden_dim[i], out_h, out_w)
            z.append(y)

            if i < self.num_layers - 1:
                x = self.repatch[i](y)
                out_h = out_h // 2
                out_w = out_w // 2

        return z


def test_ViT():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a random tensor and normalize it
    x = torch.rand(4, 3, 256, 256).to(device)

    # Create a ViT model and forward propagate the input tensor
    model = VisionTransformerMultiScale().to(device)
    outputs = model(x)

    # Print the shape of the output tensors
    for output in outputs:
        print(output.shape)


if __name__ == "__main__":
    test_ViT()
