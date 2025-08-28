import torch

from PIL import Image

from utilities import toTensor255
from datasets import CocoWikiArt
from network import VisionTransformer, AdaAttnTransformerMultiHead


MODEL_EPOCH = 20
BATCH_SIZE = 8
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

CONTENT_IDX = 66666
CONTENT_PATH = None
STYLE_PATH = None

CONTENT_PATH = "./contents/Streets.jpg"
STYLE_PATH = "./styles/Another-colorful-world.jpg"

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
    
    NUM_RUNS = 100  # Set the number of runs
    total_time = 0.0  # Initialize total time

    with torch.no_grad():
        for _ in range(NUM_RUNS):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()

            fc = vit_c(c)
            fs = vit_s(s)
            _, cs = adaFormer(fc, fs)
            cs = cs.clamp(0, 255)

            end_event.record()
            torch.cuda.synchronize()

            # Accumulate the time for each run
            total_time += start_event.elapsed_time(end_event)

    # Calculate the average time
    average_time = total_time / NUM_RUNS
    print(f"Average model inference time over {NUM_RUNS} runs: {average_time:.4f} ms")