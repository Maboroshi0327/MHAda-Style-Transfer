import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from datasets import CocoWikiArt
from lossfn import style_loss, content_loss, identity_loss_1, identity_loss_2
from network import StylizingNetwork
from vgg19 import VGG19
from vit import ViT


EPOCH_START = 1
EPOCH_END = 20
BATCH_SIZE = 8
LR = 1e-4

LAMBDA_S = 30
LAMBDA_C = 10
LAMBDA_ID1 = 1e-2
LAMBDA_ID2 = 1

ACTIAVTION = "softmax"
ENC_LAYER_NUM = 3


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Datasets
    dataloader = DataLoader(
        CocoWikiArt(),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )

    # Model
    vit_c = ViT(num_layers=ENC_LAYER_NUM, pos_embedding=True).to(device)
    vit_s = ViT(num_layers=ENC_LAYER_NUM, pos_embedding=False).to(device)
    model = StylizingNetwork(enc_layer_num=ENC_LAYER_NUM, activation=ACTIAVTION).to(device)
    vit_c.train()
    vit_s.train()
    model.train()

    # VGG19 as feature extractor (loss function)
    vgg19 = VGG19().to(device)
    vgg19.eval()

    # Loss function
    mse = nn.MSELoss(reduction="mean")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH_START, EPOCH_END + 1):

        # Batch iterator
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCH_END}", leave=True)

        # Training
        for content, style in batch_iterator:
            content = content.to(device)
            style = style.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            fc_vc = vit_c(content)
            fs_vs = vit_s(style)
            cs = model(fc_vc, fs_vs)

            fc_vs = vit_s(content)
            fs_vc = vit_c(style)
            cc = model(fc_vc, fc_vs)
            ss = model(fs_vc, fs_vs)

            # VGG19 feature extractor
            vgg_fs = vgg19(style)
            vgg_fc = vgg19(content)
            vgg_fcs = vgg19(cs)
            vgg_fcc = vgg19(cc)
            vgg_fss = vgg19(ss)

            # Style loss
            loss_s = style_loss(vgg_fcs, vgg_fs, mse)
            loss_s *= LAMBDA_S

            # Content loss
            loss_c = content_loss(vgg_fcs, vgg_fc, mse)
            loss_c *= LAMBDA_C

            # Identity loss 1
            loss_id1 = identity_loss_1(cc, content, ss, style, mse)
            loss_id1 *= LAMBDA_ID1

            # Identity loss 2
            loss_id2 = identity_loss_2(vgg_fcc, vgg_fc, vgg_fss, vgg_fs, mse)
            loss_id2 *= LAMBDA_ID2

            # Loss
            loss = loss_s + loss_c + loss_id1 + loss_id2

            # Backward pass
            loss.backward()

            optimizer.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("loss_s", loss_s.item()),
                    ("loss_c", loss_c.item()),
                    ("loss_id1", loss_id1.item()),
                    ("loss_id2", loss_id2.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

        # Save model
        torch.save(vit_c.state_dict(), f"./models/ViT_c_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")
        torch.save(vit_s.state_dict(), f"./models/ViT_s_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")
        torch.save(model.state_dict(), f"./models/AdaAttN_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")


if __name__ == "__main__":
    train()
