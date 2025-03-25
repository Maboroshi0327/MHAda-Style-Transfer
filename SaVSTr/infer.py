import torch

from utilities import toPil
from datasets import CocoWikiArt
from network import StylizingNetwork


ENC_LAYER_NUM = 3


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = CocoWikiArt()
    coco, wikiart = dataset[66666]
    print("CocoWikiArt dataset")
    print("dataset length:", len(dataset))

    model = StylizingNetwork(enc_layer_num=ENC_LAYER_NUM).to(device)
    model.load_state_dict(torch.load("./models/ViT-AdaAttN-image_epoch_5_batchSize_8.pth", weights_only=True), strict=True)
    model.eval()

    toPil(coco.byte()).save("./coco.png")
    toPil(wikiart.byte()).save("./wikiart.png")
    c, s = coco.unsqueeze(0).to(device), wikiart.unsqueeze(0).to(device)

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
