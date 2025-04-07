import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
from collections import OrderedDict

from datasets import CocoWikiArt
from lossfn import style_loss, content_loss, identity_loss_1, identity_loss_2
from network import AdaMSViT
from vgg19 import VGG19
from vit import ViT_MultiScale

# Use torchrun to launch multi-GPU training
# torchrun --nproc_per_node=2 train_AdaMSViT_multi-gpu.py

EPOCH_START = 1
EPOCH_END = 20
BATCH_SIZE = 8
LR = 1e-4

LAMBDA_S = 30
LAMBDA_C = 10
LAMBDA_ID1 = 1e-1
LAMBDA_ID2 = 1

IMAGE_SIZE = (256, 256)
ACTIAVTION = "softmax"


def train(local_rank):
    # Set the current GPU
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize the distributed process group
    dist.init_process_group(backend="nccl")

    # Datasets & DistributedSampler
    dataset = CocoWikiArt(IMAGE_SIZE)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE // torch.cuda.device_count(),
        sampler=sampler,
        num_workers=4,
        prefetch_factor=2,
    )

    # Build the model and move it to the GPU
    vit_c = ViT_MultiScale(image_size=IMAGE_SIZE, pos_embedding=True).to(device)
    vit_s = ViT_MultiScale(image_size=IMAGE_SIZE, pos_embedding=False).to(device)
    model = AdaMSViT(activation=ACTIAVTION).to(device)
    vgg19 = VGG19().to(device)

    # Wrap vit_c, vit_s and model with DDP for distributed multi-GPU training.
    vit_c = DDP(vit_c, device_ids=[local_rank])
    vit_s = DDP(vit_s, device_ids=[local_rank])
    model = DDP(model, device_ids=[local_rank])

    # Set the model to training mode or evaluation mode
    vit_c.train()
    vit_s.train()
    model.train()
    vgg19.eval()

    # Loss function & Optimizer
    mse = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=LR)
    optimizer_vit_c = optim.Adam(vit_c.parameters(), lr=LR)
    optimizer_vit_s = optim.Adam(vit_s.parameters(), lr=LR)

    # Training loop
    for epoch in range(EPOCH_START, EPOCH_END + 1):
        sampler.set_epoch(epoch)
        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCH_END}", leave=True)

        for content, style in batch_iterator:
            content = content.to(device)
            style = style.to(device)

            # Zero the gradients
            optimizer.zero_grad()
            optimizer_vit_c.zero_grad()
            optimizer_vit_s.zero_grad()

            # Forward pass for style transfer
            fc_vc = vit_c(content)
            fs_vs = vit_s(style)
            cs = model(fc_vc, fs_vs)

            # Forward pass for identity loss
            fc_vs = vit_s(content)
            fs_vc = vit_c(style)
            cc = model(fc_vc, fc_vs)
            ss = model(fs_vc, fs_vs)

            # Use VGG19 extract features for loss calculation
            vgg_fs = vgg19(style)
            vgg_fc = vgg19(content)
            vgg_fcs = vgg19(cs)
            vgg_fcc = vgg19(cc)
            vgg_fss = vgg19(ss)

            # Calculate losses
            loss_s = style_loss(vgg_fcs, vgg_fs, mse) * LAMBDA_S
            loss_c = content_loss(vgg_fcs, vgg_fc, mse) * LAMBDA_C
            loss_id1 = identity_loss_1(cc, content, ss, style, mse) * LAMBDA_ID1
            loss_id2 = identity_loss_2(vgg_fcc, vgg_fc, vgg_fss, vgg_fs, mse) * LAMBDA_ID2
            loss = loss_s + loss_c + loss_id1 + loss_id2

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer_vit_c.step()
            optimizer_vit_s.step()

            # Update the tqdm progress bar
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("loss_s", loss_s.item()),
                    ("loss_c", loss_c.item()),
                    ("loss_id1", loss_id1.item()),
                    ("loss_id2", loss_id2.item()),
                ]
            )
            batch_iterator.set_postfix(postfix)

        # Save model only on rank 0
        if dist.get_rank() == 0:
            torch.save(vit_c.module.state_dict(), f"./models/ViT_MultiScale_C_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")
            torch.save(vit_s.module.state_dict(), f"./models/ViT_MultiScale_S_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")
            torch.save(model.module.state_dict(), f"./models/AdaMSViT_epoch_{epoch}_batchSize_{BATCH_SIZE}.pth")

    # Clean up distributed process group
    dist.destroy_process_group()


if __name__ == "__main__":
    # LOCAL_RANK is automatically passed as an environment variable by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    train(local_rank)
