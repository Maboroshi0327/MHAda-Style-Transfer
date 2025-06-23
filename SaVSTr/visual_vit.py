import torch
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image

from utilities import toTensor255, toPil
from network import VisionTransformer


MODEL_EPOCH = 20
BATCH_SIZE = 8
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

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


# ---- 1. Style transfer model ----
device = "cuda" if torch.cuda.is_available() else "cpu"
vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
vit_c.load_state_dict(torch.load(VITC_PATH, map_location=device, weights_only=True), strict=True)
vit_c.requires_grad_(False)
vit_c.eval()


# ---- 2. Preparation ----
c = Image.open(CONTENT_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
c = toTensor255(c).unsqueeze(0).to(device)
s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
s = toTensor255(s).unsqueeze(0).to(device)

# Extract target features from content and style images
with torch.no_grad():
    fc = vit_c(c)
    target = fc


# ---- 3. Progressive reconstruction ----
recon = torch.randn_like(c, requires_grad=True, device=device)
optimizer = optim.Adam([recon], lr=5e-1)
num_iters = 2000

for iter in range(1, num_iters + 1):
    optimizer.zero_grad()

    # Forward pass
    fc = vit_c(recon)

    # Loss calculation
    loss = F.mse_loss(fc[0], target[0])
    loss.backward()
    optimizer.step()

    if iter % 50 == 0:
        print(f"iter {iter:3d}  loss {loss.item():.4f}")
        recon_temp = min_max_normalize_rgb(recon.detach())
        toPil(recon_temp.squeeze(0).byte()).save("./results/reconstructed_vit.jpg")

# ---- 4. Save the reconstructed image ----
recon = min_max_normalize_rgb(recon.detach())
toPil(recon.squeeze(0).byte()).save("./results/reconstructed_vit.jpg")
print(f"Saved → results/reconstructed_vit.jpg")
