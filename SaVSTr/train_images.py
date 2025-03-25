import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from collections import OrderedDict

from datasets import CocoWikiArt
from lossfn import global_stylized_loss, local_feature_loss
from network import StylizingNetwork
from vgg19 import VGG19


EPOCH_START = 1
EPOCH_END = 5
BATCH_SIZE = 8
LR = 1e-4
LAMBDA_G = 30
LAMBDA_L = 10
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
    model = StylizingNetwork(enc_layer_num=ENC_LAYER_NUM, activation=ACTIAVTION).to(device)
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
            cs = model(content, style)

            # VGG19 feature extractor
            vgg_fcs = vgg19(cs)
            vgg_fs = vgg19(style)
            vgg_fc = vgg19(content)

            # Global stylized loss
            loss_gs = 0
            loss_gs += global_stylized_loss(vgg_fcs["relu2_1"], vgg_fs["relu2_1"], mse)
            loss_gs += global_stylized_loss(vgg_fcs["relu3_1"], vgg_fs["relu3_1"], mse)
            loss_gs += global_stylized_loss(vgg_fcs["relu4_1"], vgg_fs["relu4_1"], mse)
            loss_gs += global_stylized_loss(vgg_fcs["relu5_1"], vgg_fs["relu5_1"], mse)
            loss_gs *= LAMBDA_G

            # Local feature loss
            loss_lf = 0
            loss_lf += local_feature_loss(vgg_fcs["relu3_1"], vgg_fc["relu3_1"], mse)
            loss_lf += local_feature_loss(vgg_fcs["relu4_1"], vgg_fc["relu4_1"], mse)
            loss_lf += local_feature_loss(vgg_fcs["relu5_1"], vgg_fc["relu5_1"], mse)
            loss_lf *= LAMBDA_L

            # Loss
            loss = loss_gs + loss_lf

            # Backward pass
            loss.backward()

            optimizer.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("loss_gs", loss_gs.item()),
                    ("loss_lf", loss_lf.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

        # Save model
        torch.save(model.state_dict(), f"./models/ViT-AdaAttN-image_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")


if __name__ == "__main__":
    train()
