import torch
import torch.nn as nn
from torchvision.models.optical_flow import raft_large

import cv2
from PIL import Image
from tqdm import tqdm

from utilities import toTensor255, raftTransforms, cv2_to_tensor, warp, flow_warp_mask
from network import VisionTransformer, AdaAttnTransformerMultiHead


MODEL_EPOCH = 30
BATCH_SIZE = 2
ADA_PATH = f"./models/AdaFormer_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITC_PATH = f"./models/ViT_C_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"
VITS_PATH = f"./models/ViT_S_epoch_{MODEL_EPOCH}_batchSize_{BATCH_SIZE}.pth"

VIDEO_PATH = "../datasets/Videvo/67.mp4"
STYLE_PATH = "./styles/Udnie.jpg"

IMAGE_SIZE1 = (256, 256)
IMAGE_SIZE2 = (256, 512)
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
    vit_c.load_state_dict(torch.load(VITC_PATH, weights_only=True), strict=True)
    vit_s.load_state_dict(torch.load(VITS_PATH, weights_only=True), strict=True)
    adaFormer.load_state_dict(torch.load(ADA_PATH, weights_only=True), strict=True)
    vit_c.eval()
    vit_s.eval()
    adaFormer.eval()

    # Load optical flow model
    raft = raft_large(weights="Raft_Large_Weights.C_T_SKHT_V2").to(device)
    raft = raft.eval()

    # Load style image
    s = Image.open(STYLE_PATH).convert("RGB").resize((IMAGE_SIZE1[1], IMAGE_SIZE1[0]), Image.BILINEAR)
    s = toTensor255(s).unsqueeze(0).to(device)
    with torch.no_grad():
        fs = vit_s(s)

    # Count for Warping Error & Flow MSE
    count = 0
    warping_error = 0
    flow_mse = 0
    mse = nn.MSELoss(reduction="mean")
    mseMatrix = nn.MSELoss(reduction="none")

    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}", f"FPS: {fps}")

    # Progress bar
    bar = tqdm(total=total_frames, desc="Processing video", unit="frame")

    # First frame
    frames = list()
    ret, frame = cap.read()
    frames.append(frame)
    bar.update(1)

    while True:
        ret, frame = cap.read()
        frames.append(frame)
        if not ret:
            break

        with torch.no_grad():
            # Convert frame to tensor
            c1 = cv2_to_tensor(frame[0], resize=(IMAGE_SIZE2[1], IMAGE_SIZE2[0])).unsqueeze(0).to(device)
            c2 = cv2_to_tensor(frame[1], resize=(IMAGE_SIZE2[1], IMAGE_SIZE2[0])).unsqueeze(0).to(device)

            # Forward pass
            fc1 = vit_c(c1)
            fc2 = vit_c(c2)
            _, cs1 = adaFormer(fc1, fs)
            _, cs2 = adaFormer(fc2, fs)
            cs1 = cs1.clamp(0, 255)
            cs2 = cs2.clamp(0, 255)

            # Calculate optical flow
            c1_raft = raftTransforms(c1)
            c2_raft = raftTransforms(c2)
            c_flow_01 = raft(c1_raft, c2_raft)[-1].squeeze(0)
            c_flow_10 = raft(c2_raft, c1_raft)[-1].squeeze(0)
            cs1_raft = raftTransforms(cs1)
            cs2_raft = raftTransforms(cs2)
            # cs1_flow_01 = raft(cs1_raft, cs2_raft)[-1].squeeze(0)
            cs1_flow_10 = raft(cs2_raft, cs1_raft)[-1].squeeze(0)

            # create mask
            mask = flow_warp_mask(c_flow_01, c_flow_10).unsqueeze(0)

            # Warping Error
            warped_cs1 = warp(cs1, c_flow_10)
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, cs1.shape[1], -1, -1)
            loss = torch.sum(mask * mseMatrix(cs2, warped_cs1)) / (cs1.shape[1] * cs1.shape[2] * cs1.shape[3])
            warping_error += loss
            count += 1

            # Flow MSE
            flow_mse += mse(c_flow_10, cs1_flow_10)

            # Pop the frame 1
            frames.pop(0)

            # Print loss
            warping_error_temp = torch.sqrt(warping_error / count).item()
            flow_mse_temp = (flow_mse / count).item()
            bar.set_postfix(
                warping_error=warping_error_temp,
                flow_mse=flow_mse_temp,
            )
            bar.update(1)

    bar.close()
    cap.release()
    warping_error = torch.sqrt(warping_error / count).item()
    print(f"Warping Error: {warping_error}")
    flow_mse = (flow_mse / count).item()
    print(f"Flow MSE: {flow_mse}")
