import torch

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import toTensor255, toPil
from network import StylizingNetwork
from datasets import CocoWikiArt
from vit import ViT


ENC_LAYER_NUM = 3
MODEL_EPOCH = 5
ADA_PATH = f"./models/AdaAttN_epoch_{MODEL_EPOCH}_batchSize_8.pth"
VITC_PATH = f"./models/ViT_c_epoch_{MODEL_EPOCH}_batchSize_8.pth"
VITS_PATH = f"./models/ViT_s_epoch_{MODEL_EPOCH}_batchSize_8.pth"

STYLE_PATH = "./styles/starry-night.jpg"
# STYLE_PATH = None
ACTIAVTION = "softmax"
# ACTIAVTION = "cosine"


def get_attention_hook(name):
    def hook(module, input, output):
        attentions[name] = output[0]

    return hook


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    vit_c = ViT(num_layers=ENC_LAYER_NUM, pos_embedding=True).to(device)
    vit_s = ViT(num_layers=ENC_LAYER_NUM, pos_embedding=False).to(device)
    model = StylizingNetwork(enc_layer_num=ENC_LAYER_NUM, activation=ACTIAVTION).to(device)

    vit_c.load_state_dict(torch.load(VITC_PATH, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, weights_only=True), strict=True)
    model.load_state_dict(torch.load(ADA_PATH, weights_only=True), strict=True)

    vit_c.eval()
    vit_s.eval()
    model.eval()

    # Initialize the dictionary to store attentions
    attentions = {}

    # Register hook for each transformer block in the encoder
    for idx, layer in enumerate(vit_c.encoder.layers):
        layer.self_attention.register_forward_hook(get_attention_hook(f"encoder_layer_{idx}"))

    # Load dataset
    dataset = CocoWikiArt()
    coco, wikiart = dataset[66666]

    # Use wikiart as style image if STYLE_PATH is None
    c = coco.unsqueeze(0).to(device)
    if STYLE_PATH is not None:
        s = Image.open(STYLE_PATH).convert("RGB").resize((256, 256), Image.BILINEAR)
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

    # Plot attention
    attentions = list(attentions.values())
    for idx, attn in enumerate(attentions):
        attn = attn.mean(dim=-1)
        attn = attn.view(32, 32)
        attn = attn.detach().cpu().numpy()

        plt.figure(figsize=(8, 8))
        sns.heatmap(attn, square=True, cmap="viridis")
        plt.title(f"Encoder Layer {idx} Attention")
        plt.xlabel("Token")
        plt.ylabel("Token")
        plt.savefig(f"./attention_{idx}.png")
