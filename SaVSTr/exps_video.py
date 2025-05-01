import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm

from datasets import Sintel
from utilities import toTensor255, warp
from network import VisionTransformer, AdaAttnTransformerMultiHead


# MODEL_EPOCH = 20
# BATCH_SIZE = 8
# ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
# VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
# VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

MODEL_EPOCH = 27
BATCH_SIZE = 2
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

# STYLE_PATH = "./styles/Autoportrait.png"
# STYLE_PATH = "./styles/Brushstrokes.png"
# STYLE_PATH = "./styles/Composition.png"
# STYLE_PATH = "./styles/Mosaic.png"
# STYLE_PATH = "./styles/Sketch.png"
# STYLE_PATH = "./styles/Tableau.png"
# STYLE_PATH = "./styles/The-Scream.png"
STYLE_PATH = "./styles/Udnie.png"

IMAGE_SIZE1 = (256, 256)
IMAGE_SIZE2 = (256, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"
DELTA = 20


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Datasets
    dataloader = DataLoader(
        Sintel(device=device),
        batch_size=1,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )

    # Load the model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)
    vit_c.load_state_dict(torch.load(VITC_PATH, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, weights_only=True), strict=True)
    adaFormer.load_state_dict(torch.load(ADA_PATH, weights_only=True), strict=True)
    vit_c.eval()
    vit_s.eval()
    adaFormer.eval()

    # Load style image
    s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE1[1], IMAGE_SIZE1[0]), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)
    with torch.no_grad():
        fs = vit_s(s)

    count = 0
    optical_loss = 0
    mseMatrix = nn.MSELoss(reduction="none")

    batch_iterator = tqdm(dataloader, leave=True)
    for c1, c2, flow, mask in batch_iterator:
        with torch.no_grad():
            # # Compare with previous frame
            # diff = torch.abs(c2 - c1)
            # mask_diff = diff.gt(DELTA).any(dim=1, keepdim=True)
            # mask_diff = mask_diff.expand_as(c2)
            # c2 = torch.where(mask_diff, c2, c1)

            # Forward pass
            fc1 = vit_c(c1)
            fc2 = vit_c(c2)
            _, cs1 = adaFormer(fc1, fs)
            _, cs2 = adaFormer(fc2, fs)
            cs1 = cs1.clamp(0, 255)
            cs2 = cs2.clamp(0, 255)

            # # Compare with previous output
            # diff = torch.abs(cs2 - cs1)
            # mask_diff = diff.gt(DELTA).any(dim=1, keepdim=True)
            # mask_diff = mask_diff.expand_as(cs2)
            # cs2 = torch.where(mask_diff, cs2, cs1)

            # Optical Flow Loss
            warped_cs1 = warp(cs1, flow)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, cs1.shape[1], -1, -1)
            loss = torch.sum(mask * mseMatrix(cs2, warped_cs1)) / (cs1.shape[1] * cs1.shape[2] * cs1.shape[3])
            optical_loss += loss
            count += 1

            # Print loss
            loss_temp = torch.sqrt(optical_loss).item() / count
            batch_iterator.set_postfix(optical_loss=loss_temp)

    optical_loss = torch.sqrt(optical_loss) / count
    print(f"Optical Flow Loss: {optical_loss}")
