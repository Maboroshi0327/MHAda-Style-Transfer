import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer, Encoder

from collections import OrderedDict

from utilities import imageNet1k_normalize


class ViT(VisionTransformer):
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
        self.encoder = Encoder(
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


class ViT_pretrained(nn.Module):
    def __init__(self, enc_layer_num: int = 5, pos_embedding: bool = True):
        super().__init__()
        # Load pre-trained ViT-B/16 model
        self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.vit.eval()

        # Freeze the parameters of ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Retain only the first three encoder layers for forward propagation
        self.vit.encoder.layers = self.vit.encoder.layers[:enc_layer_num]

        # Remove the positional embedding from the model
        mean = self.vit.encoder.pos_embedding.mean()
        shape = self.vit.encoder.pos_embedding.shape
        tensor = mean.expand(shape)
        if not pos_embedding:
            self.vit.encoder.pos_embedding = nn.Parameter(tensor)
            self.vit.encoder.pos_embedding.requires_grad = False

        # Create an OrderedDict to store the output of each layer
        self.outputs = OrderedDict()

        # Register the hook for each encoder layer
        for idx, block in enumerate(self.vit.encoder.layers):
            block.register_forward_hook(self.encoder_hook(f"encoder_layer_{idx}"))

    # Define the hook function
    def encoder_hook(self, name):
        def hook(model, input, output):
            self.outputs[name] = output

        return hook

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape

        # (b, c, h, w) -> (b, hidden_dim, h/16, w/16)
        x = self.vit.conv_proj(x)

        # (b, hidden_dim, h/16, w/16) -> (b, hidden_dim, 14, 14)
        x = F.interpolate(x, size=14, mode="bilinear", align_corners=False)

        # (b, hidden_dim, 14, 14) -> (b, hidden_dim, 196)
        x = x.reshape(b, self.vit.hidden_dim, 14 * 14)

        # (b, hidden_dim, 196) -> (b, 196, hidden_dim)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor) -> list:
        # Normalize the input tensor
        x = imageNet1k_normalize(x)

        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.vit.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.vit.encoder(x)

        # Remove the class token from the output
        for name, output in self.outputs.items():
            self.outputs[name] = output[:, 1:, :]

        return list(self.outputs.values())


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load a random tensor and normalize it
    x = torch.rand(4, 3, 256, 256).to(device)

    # Create a ViT model and forward propagate the input tensor
    model = ViT().to(device)
    outputs = model(x)

    # Print the shape of the output tensors
    for output in outputs:
        print(output.shape)
