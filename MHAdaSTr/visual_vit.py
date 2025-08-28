import torch
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image

from utilities import toTensor255, toPil
from network import VisionTransformer


MODEL_EPOCH = 20
BATCH_SIZE = 8
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

IMAGE_SIZE = (512, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"

IMAGE_PATH = "./styles/Woman-with-Hat.jpg"


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


# ---- 0. Device configuration ----
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 1. Style transfer model ----
vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
vit_c.load_state_dict(torch.load(VITC_PATH, map_location=device, weights_only=True), strict=True)
vit_c.requires_grad_(False)
vit_c.eval()


# # ---- 2. Preparation ----
# c = Image.open(IMAGE_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
# c = toTensor255(c).unsqueeze(0).to(device)
# # Extract target features from content and style images
# with torch.no_grad():
#     fc = vit_c(c)
#     target = fc


# # ---- 3. Progressive reconstruction ----
# num_iters = 6000

# for i in range(NUM_LAYERS):
#     recon = torch.randn_like(c, requires_grad=True, device=device)
#     optimizer = optim.Adam([recon], lr=1e-1)

#     for iter in range(1, num_iters + 1):
#         optimizer.zero_grad()

#         # Forward pass
#         fc = vit_c(recon)

#         # Loss calculation
#         layers = range(i + 1)
#         loss = sum(F.mse_loss(fc[l], target[l]) for l in layers)
#         loss.backward()
#         optimizer.step()

#         if iter % 50 == 0:
#             print(f"iter {iter:3d}  loss {loss.item():.4f}")
#             recon_temp = min_max_normalize_rgb(recon.detach())
#             toPil(recon_temp.squeeze(0).byte()).save(f"./results/reconstructed_vit_{i+1}.jpg")

#     # ---- 4. Save the reconstructed image ----
#     recon = min_max_normalize_rgb(recon.detach())
#     toPil(recon.squeeze(0).byte()).save(f"./results/reconstructed_vit_{i+1}.jpg")
#     print(f"Saved → results/reconstructed_vit_{i+1}.jpg")


# ----5. For MHAda ----
IMAGE_PATH = "./contents/Chicago.jpg"
c = Image.open(IMAGE_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
c = toTensor255(c).unsqueeze(0).to(device)

with torch.no_grad():
    fc = vit_c(c)
    target = fc

recon = torch.randn_like(c, requires_grad=True, device=device)
optimizer = optim.Adam([recon], lr=5e-1)

num_iters = 3000
for iter in range(1, num_iters + 1):
    optimizer.zero_grad()

    # Forward pass
    fc = vit_c(recon)

    # Loss calculation
    layers = range(3)
    loss = sum(F.mse_loss(fc[l], target[l]) for l in layers)
    loss.backward()
    optimizer.step()

    if iter % 50 == 0:
        print(f"iter {iter:3d}  loss {loss.item():.4f}")
        recon_temp = min_max_normalize_rgb(recon.detach())
        toPil(recon_temp.squeeze(0).byte()).save(f"./results/reconstructed_vit.jpg")

recon = min_max_normalize_rgb(recon.detach())
toPil(recon.squeeze(0).byte()).save(f"./results/reconstructed_vit.jpg")
print(f"Saved → results/reconstructed_vit.jpg")
