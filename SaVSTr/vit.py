import torch
from torchvision.models.vision_transformer import VisionTransformer, Encoder

from collections import OrderedDict


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
