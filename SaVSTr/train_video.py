import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from datasets import FlyingThings3D_Monkaa_WikiArt
from lossfn import global_style_loss, local_feature_loss, output_level_temporal_loss, feature_level_temporal_loss
from network import VisionTransformer, AdaAttnTransformerMultiHead, AdaAttnForLoss, VGG19


EPOCH_START = 21
EPOCH_END = 40
BATCH_SIZE_VIDEO = 2
BATCH_SIZE_IMAGE = 8
LR = 1e-4

LAMBDA_GS = 100
LAMBDA_LF = 15
LAMBDA_OT = 1
LAMBDA_FT = 1
LAMBDA_ID1 = 5e-2
LAMBDA_ID2 = 1e-1
# LAMBDA_GS = 70
# LAMBDA_LF = 15
# LAMBDA_OT = 1
# LAMBDA_ID1 = 5e-2
# LAMBDA_ID2 = 1e-1

IMAGE_SIZE1 = (256, 256)
IMAGE_SIZE2 = (256, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    dataloader = DataLoader(
        FlyingThings3D_Monkaa_WikiArt(IMAGE_SIZE1, IMAGE_SIZE2),
        batch_size=BATCH_SIZE_VIDEO,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )

    # Model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)
    vit_c.train()
    vit_s.train()
    adaFormer.train()

    # AdaAttN for calculating local feature loss
    adaattnNoLearn = nn.ModuleList(
        [
            AdaAttnForLoss(256, 64 + 128 + 256, ACTIAVTION),
            AdaAttnForLoss(512, 64 + 128 + 256 + 512, ACTIAVTION),
            AdaAttnForLoss(512, 64 + 128 + 256 + 512 + 512, ACTIAVTION),
        ]
    )
    adaattnNoLearn = adaattnNoLearn.to(device)
    adaattnNoLearn.eval()

    # VGG19 as feature extractor (loss function)
    vgg19 = VGG19().to(device)
    vgg19.eval()

    # Loss function
    mse = nn.MSELoss(reduction="mean")
    mseMatrix = nn.MSELoss(reduction="none")

    # Optimizer
    optimizer_vit_c = optim.Adam(vit_c.parameters(), lr=LR)
    optimizer_vit_s = optim.Adam(vit_s.parameters(), lr=LR)
    optimizer_ada = optim.Adam(adaFormer.parameters(), lr=LR)

    # Load checkpoint
    if EPOCH_START > 1:
        checkpoint = torch.load(f"./models/checkpoint_epoch_{EPOCH_START - 1}_batchSize_{BATCH_SIZE_IMAGE}.pth", map_location=device, weights_only=True)

        adaFormer.load_state_dict(checkpoint["model_state"]["adaFormer"])
        vit_c.load_state_dict(checkpoint["model_state"]["vit_c"])
        vit_s.load_state_dict(checkpoint["model_state"]["vit_s"])

        optimizer_ada.load_state_dict(checkpoint["optim_state"]["adaFormer"])
        optimizer_vit_c.load_state_dict(checkpoint["optim_state"]["vit_c"])
        optimizer_vit_s.load_state_dict(checkpoint["optim_state"]["vit_s"])

    # Training loop
    for epoch in range(EPOCH_START, EPOCH_END + 1):

        # Batch iterator
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCH_END}", leave=True)

        # Training
        for idx, (style, c1, c2, flow, mask) in enumerate(batch_iterator):
            style = style.to(device)
            c1 = c1.to(device)
            c2 = c2.to(device)
            flow = flow.to(device)
            mask = mask.to(device)

            # Zero the gradients
            optimizer_vit_c.zero_grad()
            optimizer_vit_s.zero_grad()
            optimizer_ada.zero_grad()

            # Forward pass
            vitc_fc1 = vit_c(c1)
            vitc_fc2 = vit_c(c2)
            vits_fs = vit_s(style)
            ada_fcs1, cs1 = adaFormer(vitc_fc1, vits_fs)
            ada_fcs2, cs2 = adaFormer(vitc_fc2, vits_fs)

            vits_fc1 = vit_s(c1)
            vits_fc2 = vit_s(c2)
            vitc_fs = vit_c(style)
            _, cc1 = adaFormer(vitc_fc1, vits_fc1)
            _, cc2 = adaFormer(vitc_fc2, vits_fc2)
            _, ss = adaFormer(vitc_fs, vits_fs)

            # VGG19 feature extractor
            with torch.no_grad():
                vgg_fc1 = vgg19(c1)
                vgg_fc2 = vgg19(c2)
                vgg_fs = vgg19(style)
            vgg_fcs1 = vgg19(cs1)
            vgg_fcs2 = vgg19(cs2)

            vgg_fcc1 = vgg19(cc1)
            vgg_fcc2 = vgg19(cc2)
            vgg_fss = vgg19(ss)

            # Global Style Loss
            loss_gs = global_style_loss(vgg_fcs1, vgg_fs, mse)
            loss_gs += global_style_loss(vgg_fcs2, vgg_fs, mse)
            loss_gs *= LAMBDA_GS

            # Local Feature Loss
            loss_lf = local_feature_loss(vgg_fc1, vgg_fs, vgg_fcs1, adaattnNoLearn, mse)
            loss_lf += local_feature_loss(vgg_fc2, vgg_fs, vgg_fcs2, adaattnNoLearn, mse)
            loss_lf *= LAMBDA_LF

            # Output-Level Temporal Loss
            loss_ot = output_level_temporal_loss(c1, c2, cs1, cs2, flow, mask, mseMatrix)
            loss_ot *= LAMBDA_OT

            # Feature-Level Temporal Loss
            loss_ft = feature_level_temporal_loss(ada_fcs1, ada_fcs2, flow, mask, mseMatrix)
            loss_ft *= LAMBDA_FT

            # Identity Loss 1
            loss_id1 = mse(cc1, c1) + mse(cc2, c2) + mse(ss, style)
            loss_id1 *= LAMBDA_ID1

            # Identity Loss 2
            loss_id2 = 0
            for i in [1, 2, 3, 4, 5]:
                loss_id2 += mse(vgg_fcc1[f"relu{i}_1"], vgg_fc1[f"relu{i}_1"])
                loss_id2 += mse(vgg_fcc2[f"relu{i}_1"], vgg_fc2[f"relu{i}_1"])
                loss_id2 += mse(vgg_fss[f"relu{i}_1"], vgg_fs[f"relu{i}_1"])
            loss_id2 *= LAMBDA_ID2

            # Loss
            loss = loss_gs + loss_lf + loss_ot + loss_ft + loss_id1 + loss_id2

            # Backward pass
            loss.backward()

            # Update weights
            optimizer_vit_c.step()
            optimizer_vit_s.step()
            optimizer_ada.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("l_gs", loss_gs.item()),
                    ("l_lf", loss_lf.item()),
                    ("l_ot", loss_ot.item()),
                    ("l_ft", loss_ft.item()),
                    ("l_id1", loss_id1.item()),
                    ("l_id2", loss_id2.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

            if idx % 100 == 0:
                # Save model weights
                torch.save(adaFormer.state_dict(), "./models/AdaFormer.pth")
                torch.save(vit_c.state_dict(), "./models/ViT_C.pth")
                torch.save(vit_s.state_dict(), "./models/ViT_S.pth")

        # Save model
        torch.save(adaFormer.state_dict(), f"./models/AdaFormer_epoch_{epoch}_batchSize_{BATCH_SIZE_VIDEO}.pth")
        torch.save(vit_c.state_dict(), f"./models/ViT_C_epoch_{epoch}_batchSize_{BATCH_SIZE_VIDEO}.pth")
        torch.save(vit_s.state_dict(), f"./models/ViT_S_epoch_{epoch}_batchSize_{BATCH_SIZE_VIDEO}.pth")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "batch_size": BATCH_SIZE_VIDEO,
            "model_state": {
                "adaFormer": adaFormer.state_dict(),
                "vit_c": vit_c.state_dict(),
                "vit_s": vit_s.state_dict(),
            },
            "optim_state": {
                "adaFormer": optimizer_ada.state_dict(),
                "vit_c": optimizer_vit_c.state_dict(),
                "vit_s": optimizer_vit_s.state_dict(),
            },
        }
        torch.save(checkpoint, f"./models/checkpoint_epoch_{epoch}_batchSize_{BATCH_SIZE_VIDEO}.pth")


if __name__ == "__main__":
    train()
