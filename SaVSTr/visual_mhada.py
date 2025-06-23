import torch
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image

from utilities import toTensor255, toPil
from network import VisionTransformer
from network.adaDecoder import AdaAttnMultiHead, Decoder


MODEL_EPOCH = 20
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

IMAGE_SIZE = (256, 256)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"

CONTENT_PATH = "./contents/Chair.jpg"
STYLE_PATH = "./styles/Sketch.jpg"


def min_max_normalize(tensor):
    """Normalize a tensor to the range [0, 255]"""
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val) * 255.0


def min_max_normalize_rgb(tensor):
    """Normalize a 3-channel RGB tensor to the range [0, 255]"""
    tensor[0, 0, :, :] = min_max_normalize(tensor[0, 0, :, :])
    tensor[0, 1, :, :] = min_max_normalize(tensor[0, 1, :, :])
    tensor[0, 2, :, :] = min_max_normalize(tensor[0, 2, :, :])
    return tensor


class AdaAttnTransformerMultiHead(nn.Module):
    def __init__(self, num_layers: int = 3, qkv_dim: int = 512, num_heads: int = 8, activation: str = "softmax"):
        super().__init__()
        self.num_layers = num_layers

        self.adaAttnHead = nn.ModuleList(
            [
                AdaAttnMultiHead(
                    qkv_dim=qkv_dim,
                    num_heads=num_heads,
                    activation=activation,
                )
                for _ in range(num_layers * 2)
            ]
        )

        self.decoder = Decoder()

    def forward(self, *args):
        # Support two calling conventions:
        #   1) model(fc_list, fs_list)
        #   2) model((fc_list, fs_list))   ← ptflops passes arguments this way
        if len(args) == 1:
            fc, fs = args[0]
        else:
            fc, fs = args

        fcs = fc[0]
        for i in range(self.num_layers):
            fcs = self.adaAttnHead[2 * i](fc[i], fs[i], fcs)
            fcs = self.adaAttnHead[2 * i + 1](fcs, fs[i], fcs)

        return fcs


# ---- 1. Style transfer model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)

vit_c.load_state_dict(torch.load(VITC_PATH, map_location=device, weights_only=True), strict=True)
vit_s.load_state_dict(torch.load(VITS_PATH, map_location=device, weights_only=True), strict=True)
adaFormer.load_state_dict(torch.load(ADA_PATH, map_location=device, weights_only=True), strict=True)

vit_c.requires_grad_(False)
vit_s.requires_grad_(False)
adaFormer.requires_grad_(False)

vit_c.eval()
vit_s.eval()
adaFormer.eval()


# ---- 2. Preparation ----
c = Image.open(CONTENT_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
c = toTensor255(c).unsqueeze(0).to(device)
s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
s = toTensor255(s).unsqueeze(0).to(device)

# Extract target features from content and style images
with torch.no_grad():
    fc = vit_c(c)
    fs = vit_s(s)
    fcs = adaFormer(fc, fs)
    target = fcs


# ---- 3. Progressive reconstruction ----
recon = torch.randn_like(c, requires_grad=True, device=device)
optimizer = optim.Adam([recon], lr=5e-1)
num_iters = 1500

for iter in range(1, num_iters + 1):
    optimizer.zero_grad()

    # Forward pass
    fc = vit_c(recon)
    fs = vit_s(s)
    fcs = adaFormer(fc, fs)

    # Loss calculation
    loss = F.mse_loss(fcs, target)
    loss.backward()
    optimizer.step()

    if iter % 50 == 0:
        print(f"iter {iter:3d}  loss {loss.item():.4f}")
        recon_temp = min_max_normalize_rgb(recon.detach())
        toPil(recon_temp.squeeze(0).byte()).save("./results/reconstructed_mhada.jpg")

# ---- 4. Save the reconstructed image ----
recon = min_max_normalize_rgb(recon.detach())
toPil(recon.squeeze(0).byte()).save("./results/reconstructed_mhada.jpg")
print(f"Saved → results/reconstructed_mhada.jpg")
