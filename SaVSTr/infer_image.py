import torch

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import toTensor255, toPil
from datasets import CocoWikiArt
from network import VisionTransformer, AdaAttnTransformerMultiHead


MODEL_EPOCH = 20
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

# ADA_PATH = "./models/AdaFormer.pth"
# VITC_PATH = "./models/ViT_C.pth"
# VITS_PATH = "./models/ViT_S.pth"

CONTENT_IDX = 66666
CONTENT_PATH = None
STYLE_PATH = None

CONTENT_PATH = "./contents/Bair.png"
STYLE_PATH = "./styles/Installation-at-ACE-Gallery-New-York.jpg"

IMAGE_SIZE = (512, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=True).to(device)
    vit_s = VisionTransformer(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, hidden_dim=HIDDEN_DIM, pos_embedding=False).to(device)
    adaFormer = AdaAttnTransformerMultiHead(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, qkv_dim=HIDDEN_DIM, activation=ACTIAVTION).to(device)

    vit_c.load_state_dict(torch.load(VITC_PATH, map_location=device, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, map_location=device, weights_only=True), strict=True)
    adaFormer.load_state_dict(torch.load(ADA_PATH, map_location=device, weights_only=True), strict=True)

    vit_c.eval()
    vit_s.eval()
    adaFormer.eval()

    # Load dataset
    dataset = CocoWikiArt(IMAGE_SIZE)
    coco, wikiart = dataset[CONTENT_IDX]

    # Use COCO as content image if CONTENT_PATH is None
    if CONTENT_PATH is not None:
        c = Image.open(CONTENT_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        c = toTensor255(c).unsqueeze(0).to(device)
    else:
        c = coco.unsqueeze(0).to(device)

    # Use wikiart as style image if STYLE_PATH is None
    if STYLE_PATH is not None:
        s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)
    else:
        s = wikiart.unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        fc = vit_c(c)
        fs = vit_s(s)
        _, cs = adaFormer(fc, fs)
        cs = cs.clamp(0, 255)
        print(cs[0, 0].min(), cs[0, 0].max())
        print(cs[0, 1].min(), cs[0, 1].max())
        print(cs[0, 2].min(), cs[0, 2].max())

    # Save images
    toPil(c.squeeze(0).byte()).save("./results/content.png")
    toPil(s.squeeze(0).byte()).save("./results/style.png")
    toPil(cs.squeeze(0).byte()).save("./results/stylized.png")

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
        plt.savefig(f"./results/attention_c_{idx}.png")

    for idx, feat in enumerate(fs):
        b, c, h, w = feat.shape
        feat = feat.mean(dim=1)
        feat = feat.view(h, w)
        feat = feat.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        sns.heatmap(feat, square=True, cmap="viridis")
        plt.title(f"Feature Maps {idx + 1}")
        plt.xlabel("Token")
        plt.ylabel("Token")
        plt.savefig(f"./results/attention_s_{idx}.png")
