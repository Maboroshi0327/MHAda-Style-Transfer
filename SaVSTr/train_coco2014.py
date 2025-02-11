import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from datasets import Coco2014, toTensor255
from temp import TextureNetworks, Vgg16
from utilities import gram_matrix, vgg_normalize


device = "cuda" if torch.cuda.is_available() else "cpu"
epoch_start = 1
epoch_end = 10
batch_size = 4
LR = 1e-3
ALPHA = 1e5
BETA = 1e10
IMG_SIZE = (256, 256)


def train():
    # Datasets and model
    dataloader = DataLoader(
        Coco2014("../datasets/coco2014", IMG_SIZE),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2,
    )
    model = TextureNetworks().to(device)

    # Optimizer and loss
    adam = optim.Adam(model.parameters(), lr=LR)
    L2distance = nn.MSELoss(reduction="mean")
    vgg16 = Vgg16().to(device)

    # Style image
    style_img_path = "./styles/mosaic.jpg"
    style = Image.open(style_img_path).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    style = toTensor255(style).unsqueeze(0).to(device)

    # Style image Gram Matrix
    style_features = vgg16(vgg_normalize(style))
    style_GM = [gram_matrix(f) for f in style_features]

    # Training loop
    for epoch in range(epoch_start, epoch_end + 1):
        model.train()

        batch_iterator = tqdm(dataloader, desc=f"Epoch {epoch}/{epoch_end}", leave=True)
        for img in batch_iterator:
            img = img.to(device)

            # Zero gradients
            adam.zero_grad()

            # Forward pass
            styled_img = model(img)

            # Normalize and use VGG16 to get features
            styled_img = vgg_normalize(styled_img)
            img = vgg_normalize(img)
            styled_features = vgg16(styled_img)
            content_features1 = vgg16(img)

            # Content Loss
            content_loss = 0
            content_loss += L2distance(styled_features[2], content_features1[2])
            content_loss *= ALPHA

            # Style Loss
            style_loss = 0
            for i, gram_s in enumerate(style_GM):
                gram_img1 = gram_matrix(styled_features[i])
                style_loss += L2distance(gram_img1, gram_s.expand(gram_img1.shape[0], -1, -1))
            style_loss *= BETA

            # Total Loss
            loss = content_loss + style_loss

            # Backward pass
            loss.backward()
            adam.step()

            # Use OrderedDict to set suffix information
            postfix = OrderedDict(
                [
                    ("loss", loss.item()),
                    ("SL", style_loss.item()),
                    ("CL", content_loss.item()),
                ]
            )

            # Update progress bar
            batch_iterator.set_postfix(postfix)

        # Save model
        torch.save(model.state_dict(), f"./models/Coco2014_epoch_{epoch}_batchSize_{batch_size}.pth")


if __name__ == "__main__":
    train()
