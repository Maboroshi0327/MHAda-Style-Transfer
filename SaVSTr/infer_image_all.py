import torch

from PIL import Image

from utilities import toTensor255, toPil, list_files
from network import VisionTransformer, AdaAttnTransformerMultiHead


MODEL_EPOCH = 20
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

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

    # Load images
    print("Loading images...")
    c = list()
    s = list()
    for content_path in list_files("./contents"):
        img = Image.open(content_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        img = toTensor255(img).unsqueeze(0).to(device)
        c.append(img)

    for style_path in list_files("./styles"):
        img = Image.open(style_path).convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        img = toTensor255(img).unsqueeze(0).to(device)
        s.append(img)

    # Model inference
    for i, content in enumerate(c):
        for j, style in enumerate(s):
            print(f"Processing content {i + 1} and style {j + 1}...")

            # Model inference
            with torch.no_grad():
                fc = vit_c(content)
                fs = vit_s(style)
                _, cs = adaFormer(fc, fs)
                cs = cs.clamp(0, 255)

            # Save the results
            save_path = f"./results/content_{i + 1}_style_{j + 1}.jpg"
            toPil(cs[0].byte()).save(save_path)
