import torch

from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

from utilities import toTensor255, toPil
from network import StylizingNetwork
from datasets import CocoWikiArt


MODEL_PATH = "./models/ViT-AdaAttN-image_epoch_2_batchSize_8.pth"
STYLE_PATH = "./styles/starry-night.jpg"
# STYLE_PATH = None
ACTIAVTION = "softmax"
# ACTIAVTION = "cosine"
ENC_LAYER_NUM = 3


def get_attention_hook(name):
    def hook(module, input, output):
        attentions[name] = output[0]

    return hook


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    model = StylizingNetwork(enc_layer_num=ENC_LAYER_NUM, activation=ACTIAVTION).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True), strict=True)
    model.eval()

    # Initialize the dictionary to store attentions
    attentions = {}

    # Register hook for each transformer block in the encoder
    for idx, layer in enumerate(model.vit_c.encoder.layers):
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
        cs = model(c, s)
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
