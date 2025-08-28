import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from datasets import CocoWikiArt
from lossfn import global_style_loss, local_feature_loss, identity_loss_1, identity_loss_2
from network import VisionTransformer, AdaAttnTransformerMultiHead, AdaAttnForLoss, VGG19


EPOCH_START = 1
EPOCH_END = 20
BATCH_SIZE = 8
LR = 1e-4

LAMBDA_GS = 70
LAMBDA_LF = 15
LAMBDA_ID1 = 5e-2
LAMBDA_ID2 = 1e-1

IMAGE_SIZE = (256, 256)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    dataloader = DataLoader(
        CocoWikiArt(IMAGE_SIZE),
        batch_size=BATCH_SIZE,
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

    # Optimizer
    optimizer_vit_c = optim.Adam(vit_c.parameters(), lr=LR)
    optimizer_vit_s = optim.Adam(vit_s.parameters(), lr=LR)
    optimizer_ada = optim.Adam(adaFormer.parameters(), lr=LR)

    # Load checkpoint
    if EPOCH_START > 1:
        checkpoint = torch.load(f"./models/checkpoint_epoch_{EPOCH_START - 1}_batchSize_{BATCH_SIZE}.pth", map_location=device, weights_only=True)

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
        for idx, (content, style) in enumerate(batch_iterator):
            content = content.to(device)
            style = style.to(device)

            # Zero the gradients
            optimizer_vit_c.zero_grad()
            optimizer_vit_s.zero_grad()
            optimizer_ada.zero_grad()

            # Forward pass
            fc_vc = vit_c(content)
            fs_vs = vit_s(style)
            _, cs = adaFormer(fc_vc, fs_vs)

            fc_vs = vit_s(content)
            fs_vc = vit_c(style)
            _, cc = adaFormer(fc_vc, fc_vs)
            _, ss = adaFormer(fs_vc, fs_vs)

            # VGG19 feature extractor
            vgg_fs = vgg19(style)
            vgg_fc = vgg19(content)
            vgg_fcs = vgg19(cs)
            vgg_fcc = vgg19(cc)
            vgg_fss = vgg19(ss)

            # Global Style Loss
            loss_gs = global_style_loss(vgg_fcs, vgg_fs, mse)
            loss_gs *= LAMBDA_GS

            # Local Feature Loss
            loss_lf = local_feature_loss(vgg_fc, vgg_fs, vgg_fcs, adaattnNoLearn, mse)
            loss_lf *= LAMBDA_LF

            # Identity Loss 1
            loss_id1 = identity_loss_1(cc, content, ss, style, mse)
            loss_id1 *= LAMBDA_ID1

            # Identity Loss 2
            loss_id2 = identity_loss_2(vgg_fcc, vgg_fc, vgg_fss, vgg_fs, mse)
            loss_id2 *= LAMBDA_ID2

            # Loss
            loss = loss_gs + loss_lf + loss_id1 + loss_id2

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
                    ("loss_gs", loss_gs.item()),
                    ("loss_lf", loss_lf.item()),
                    ("loss_id1", loss_id1.item()),
                    ("loss_id2", loss_id2.item()),
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
        torch.save(adaFormer.state_dict(), f"./models/AdaFormer_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")
        torch.save(vit_c.state_dict(), f"./models/ViT_C_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")
        torch.save(vit_s.state_dict(), f"./models/ViT_S_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "batch_size": BATCH_SIZE,
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
        torch.save(checkpoint, f"./models/checkpoint_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")


if __name__ == "__main__":
    train()
