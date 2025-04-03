import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer as VisionTransformer_torch
from torchvision.models.vision_transformer import Encoder as Encoder_torch

from collections import OrderedDict


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


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        scale: list,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale = scale

        self.encoder_layer_1 = EncoderBlock(num_heads, hidden_dim, mlp_dim)
        self.encoder_layer_2 = EncoderBlock(num_heads, hidden_dim, mlp_dim)
        self.encoder_layer_3 = EncoderBlock(num_heads, hidden_dim, mlp_dim)

        self.conv_proj = nn.ModuleList(
            [
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=2, stride=2),
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=2, stride=2),
            ]
        )

    def repatch(self, x: torch.Tensor, idx: int):
        x = x.permute(0, 2, 1).view(-1, self.hidden_dim, self.scale[idx][0], self.scale[idx][1])
        x = self.conv_proj[idx](x)
        x = x.view(-1, self.hidden_dim, self.scale[idx + 1][0] * self.scale[idx + 1][1]).permute(0, 2, 1)
        return x

    def forward(self, x: torch.Tensor):
        x1 = self.encoder_layer_1(x)
        x = self.repatch(x1, 0)
        x2 = self.encoder_layer_2(x)
        x = self.repatch(x2, 1)
        x3 = self.encoder_layer_3(x)
        return x1, x2, x3


class ViT_MultiScale(nn.Module):
    def __init__(
        self,
        image_size: tuple = (256, 256),
        patch_size: int = 4,
        num_heads: int = 8,
        hidden_dim: int = 512,
        mlp_dim: int = 2048,
        pos_embedding: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.patch_height = image_size[0] // patch_size
        self.patch_width = image_size[1] // patch_size

        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)

        if pos_embedding:
            seq_length = self.patch_height * self.patch_width
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        else:
            self.pos_embedding = 0

        scale = list()
        for i in range(3):
            scale.append((self.patch_height // (2**i), self.patch_width // (2**i)))

        self.encoder = Encoder(num_heads, hidden_dim, mlp_dim, scale)

    def forward(self, x: torch.Tensor):
        x = self.conv_proj(x)
        x = x.view(-1, self.hidden_dim, self.patch_height * self.patch_width)
        x = x.permute(0, 2, 1)

        x = x + self.pos_embedding

        x1, x2, x3 = self.encoder(x)
        return x1, x2, x3


class ViT_torch(VisionTransformer_torch):
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 8,
        num_layers: int = 3,
        num_heads: int = 8,
        hidden_dim: int = 512,
        mlp_dim: int = 2048,
        pos_embedding: bool = True,
    ):
        super().__init__(image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim)

        self.class_token = None
        seq_length = (image_size // patch_size) ** 2
        self.encoder = Encoder_torch(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            self.dropout,
            self.attention_dropout,
            self.norm_layer,
        )
        self.seq_length = seq_length
        self.heads = None

        if not pos_embedding:
            self.encoder._parameters.pop("pos_embedding", None)
            self.encoder.pos_embedding = 0

        self.outputs = OrderedDict()
        for idx, block in enumerate(self.encoder.layers):
            block.register_forward_hook(self.encoder_hook(f"encoder_layer_{idx}"))

    def encoder_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output

        return hook

    def forward(self, x: torch.Tensor):
        x = self._process_input(x)
        x = self.encoder(x)
        return list(self.outputs.values())


def test_vit_multiscale():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a random tensor and normalize it
    x = torch.rand(4, 3, 256, 256).to(device)

    # Create a ViT model and forward propagate the input tensor
    model = ViT_MultiScale().to(device)
    x1, x2, x3 = model(x)

    # Print the shape of the output tensors
    print(x1.shape)
    print(x2.shape)
    print(x3.shape)


def test_vit_torch():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a random tensor and normalize it
    x = torch.rand(4, 3, 256, 256).to(device)

    # Create a ViT model and forward propagate the input tensor
    model = ViT_torch().to(device)
    outputs = model(x)

    # Print the shape of the output tensors
    for output in outputs:
        print(output.shape)


if __name__ == "__main__":
    # test_vit_torch()
    test_vit_multiscale()
