import torch

from PIL import Image

from utilities import toTensor255, toPil
from datasets import CocoWikiArt
from network import StylizingNetwork


MODEL_PATH = "./models/ViT-AdaAttN-image_epoch_5_batchSize_8.pth"
STYLE_PATH = "./styles/starry-night.jpg"
# STYLE_PATH = None
ACTIAVTION = "softmax"
# ACTIAVTION = "cosine"
ENC_LAYER_NUM = 3


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CocoWikiArt()
    coco, wikiart = dataset[66666]

    model = StylizingNetwork(enc_layer_num=ENC_LAYER_NUM, activation=ACTIAVTION).to(device)
    model.load_state_dict(torch.load("./models/ViT-AdaAttN-image_epoch_5_batchSize_8.pth", weights_only=True), strict=True)
    model.eval()

    c = coco.unsqueeze(0).to(device)
    if STYLE_PATH is not None:
        s = Image.open(STYLE_PATH).convert("RGB").resize((256, 256), Image.BILINEAR)
        s = toTensor255(s).unsqueeze(0).to(device)
    else:
        s = wikiart.unsqueeze(0).to(device)

    toPil(c.squeeze(0).byte()).save("./content.png")
    toPil(s.squeeze(0).byte()).save("./style.png")

    with torch.no_grad():
        cs = model(c, s)
        cs = cs.clamp(0, 255)

        min_v = cs.min()
        max_v = cs.max()
        print(min_v, max_v)
        # cs = (cs - min_v) / (max_v - min_v) * 255

        cs = cs.squeeze(0)
        cs = toPil(cs.byte())
        cs.save("./stylized.png")
