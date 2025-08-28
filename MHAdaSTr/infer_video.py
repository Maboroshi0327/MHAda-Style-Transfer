import torch

import os
import cv2
from PIL import Image
import imageio
import numpy as np

from utilities import toTensor255, cv2_to_tensor, mkdir
from network import VisionTransformer, AdaAttnTransformerMultiHead


# MODE = "Original"
MODE = "Stylized"

MODEL_EPOCH = 30
BATCH_SIZE = 2
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

# VIDEO_PATH = "../datasets/Videvo/67.mp4"
# STYLE_PATH = "./styles/Autoportrait.jpg"

VIDEO_PATH = "../datasets/Videvo/19.mp4"
STYLE_PATH = "./styles/Sketch.jpg"

# VIDEO_PATH = "../datasets/Videvo/98.mp4"
# STYLE_PATH = "./styles/The-Scream.jpg"

# VIDEO_PATH = "../datasets/Videvo/38.mp4"
# STYLE_PATH = "./styles/Untitled-1964.jpg"

IMAGE_SIZE1 = (256, 256)
IMAGE_SIZE2 = (256, 512)
NUM_LAYERS = 3
NUM_HEADS = 8
HIDDEN_DIM = 512
ACTIAVTION = "softmax"
DELTA = 20


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

    # Load style image
    s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE1[1], IMAGE_SIZE1[0]), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)
    with torch.no_grad():
        fs = vit_s(s)

    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    # Temporarily store video frames
    frames = list()
    prev_c = None
    prev_cs = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        with torch.no_grad():
            # Convert frame to tensor
            c = cv2_to_tensor(frame, resize=(IMAGE_SIZE2[1], IMAGE_SIZE2[0]))
            c = c.unsqueeze(0).to(device)

            # Compare with previous frame
            # if prev_c is not None:
            #     diff = torch.abs(c - prev_c)
            #     mask = diff.gt(DELTA).any(dim=1, keepdim=True)
            #     mask = mask.expand_as(c)
            #     c = torch.where(mask, c, prev_c)

            # Forward pass
            fc = vit_c(c)
            _, cs = adaFormer(fc, fs)
            if MODE == "Stylized":
                cs = cs.clamp(0, 255)
            else:
                cs = c.clamp(0, 255)

            # Compare with previous output
            # if prev_cs is not None:
            #     diff = torch.abs(cs - prev_cs)
            #     mask = diff.gt(DELTA).any(dim=1, keepdim=True)
            #     mask = mask.expand_as(cs)
            #     cs = torch.where(mask, cs, prev_cs)

            # Update previous frames
            prev_c = c
            prev_cs = cs

        # Convert output tensor back to image format
        cs = cs.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        frames.append(cs)
        cs = cv2.cvtColor(cs, cv2.COLOR_RGB2BGR)

        # Display the frame
        cv2.imshow("Frames", cs)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    # Create output directory if it doesn't exist
    SAVE_PATH = "./results/Video"
    mkdir(SAVE_PATH, delete_existing_files=True)

    # Save the frames as images
    for i in range(len(frames)):
        imageio.imwrite(os.path.join(SAVE_PATH, f"{MODE}_{i}.jpg"), frames[i])

    # Save the frames as a video
    imageio.mimsave(os.path.join(SAVE_PATH, f"{MODE}.mp4"), frames, fps=fps)
