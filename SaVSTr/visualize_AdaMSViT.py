import torch

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import toTensor255, toPil
from datasets import CocoWikiArt
from vit import ViT_MultiScale
from network import AdaMSViT


MODEL_EPOCH = 12
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaMSViT_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_MultiScale_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_MultiScale_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

CONTENT_IDX = 66666
STYLE_PATH = "./styles/candy.jpg"
# STYLE_PATH = None

IMAGE_SIZE = (256, 256)
ACTIAVTION = "softmax"
# ACTIAVTION = "cosine"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = ViT_MultiScale(image_size=IMAGE_SIZE, pos_embedding=True).to(device)
    vit_s = ViT_MultiScale(image_size=IMAGE_SIZE, pos_embedding=False).to(device)
    model = AdaMSViT(activation=ACTIAVTION).to(device)

    vit_c.load_state_dict(torch.load(VITC_PATH, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, weights_only=True), strict=True)
    model.load_state_dict(torch.load(ADA_PATH, weights_only=True), strict=True)

    vit_c.eval()
    vit_s.eval()
    model.eval()

    # Load dataset
    dataset = CocoWikiArt(IMAGE_SIZE)
    coco, wikiart = dataset[CONTENT_IDX]

    # Use wikiart as style image if STYLE_PATH is None
    c = coco.unsqueeze(0).to(device)
    if STYLE_PATH is not None:
        pil_size = (IMAGE_SIZE[1], IMAGE_SIZE[0])
        s = Image.open(STYLE_PATH).convert("RGB").resize(pil_size, Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)
    else:
        s = wikiart.unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        fc = vit_c(c)
        fs = vit_s(s)
        cs = model(fc, fs)
        cs = cs.clamp(0, 255)

    # Save images
    toPil(c.squeeze(0).byte()).save("./content.png")
    toPil(s.squeeze(0).byte()).save("./style.png")
    toPil(cs.squeeze(0).byte()).save("./stylized.png")

    # Plot Feature Maps
    for idx, feat in enumerate(fc):
        b, c, h, w = feat.shape
        feat = feat.mean(dim=1)
        feat = feat.view(h, w)
        feat = feat.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        sns.heatmap(feat, square=True, cmap="viridis")
        plt.title(f"Feature Maps {idx + 1}")
        plt.xlabel("Token")
        plt.ylabel("Token")
        plt.savefig(f"./attention_{idx}.png")
