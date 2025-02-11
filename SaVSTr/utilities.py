import os
from typing import Union

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

toTensor255 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)
toTensor = transforms.ToTensor()
toPil = transforms.ToPILImage()
gaussianBlur = transforms.GaussianBlur(kernel_size=3, sigma=1.0)


def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return sorted(files)


def visualize_flow(flow):
    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[0].cpu().numpy(), flow[1].cpu().numpy())
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb


def warp(x, flo, padding_mode="zeros"):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, mode="bilinear", padding_mode=padding_mode, align_corners=False)
    return output


def flow_warp_mask(flo01, flo10, padding_mode="zeros"):
    flo01 = flo01.unsqueeze(0)
    flo10 = flo10.unsqueeze(0)
    B, C, H, W = flo01.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    if flo01.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo10
    flo01 = grid + flo01

    # Scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)
    flow_warp = F.grid_sample(flo01, vgrid, mode="bilinear", padding_mode=padding_mode, align_corners=False)

    # create mask
    flow_warp = flow_warp.squeeze(0)
    grid = grid.squeeze(0)
    warp_error = torch.abs(flow_warp - grid)
    warp_error = torch.sum(warp_error, dim=0)
    mask = warp_error < 2
    mask = mask.float()

    return mask


def gram_matrix(y: torch.Tensor):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def vgg_normalize(batch: torch.Tensor):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std


def print_parameters(model):
    for name, _ in model.named_parameters():
        print(name)


def print_state_dict(state_dict):
    for name, _ in state_dict.items():
        print(name)


def cvframe_to_tensor(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if frame.shape != (360, 640, 3):
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
    return toTensor255(frame)


def calculate_mse(model_class, input_frame_num: int, model_path: str, video_path: str, device: str = "cuda"):
    model = model_class(input_frame_num).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    cap = cv2.VideoCapture(video_path)
    mse = nn.MSELoss(reduction="mean")
    content_styled_imgs = list()
    count = 0
    loss = 0

    # Read the few frames to match the input_frame_num
    imgs = list()
    for _ in range(input_frame_num):
        _, frame = cap.read()
        imgs.append(cvframe_to_tensor(frame))

    # Start calculating the MSE
    while True:
        # Pass the input tensor through the model
        with torch.no_grad():
            input_tensor = torch.cat(imgs, dim=0).unsqueeze(0).to(device)
            *_, output_tensor = model(input_tensor)
            output_tensor = output_tensor.clamp(0, 255)

        content_img = imgs[-1].unsqueeze(0).to(device)
        styled_img = output_tensor
        content_styled_imgs.append([content_img, styled_img])

        if len(content_styled_imgs) == 2:
            x_t = content_styled_imgs[0][0]
            x_t1 = content_styled_imgs[1][0]
            y_t = content_styled_imgs[0][1]
            y_t1 = content_styled_imgs[1][1]

            x = x_t1 - x_t
            y = y_t1 - y_t
            loss += mse(x, y).item()

            count += 1
            content_styled_imgs.pop(0)

        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Update the input tensor list
        imgs.pop(0)
        imgs.append(cvframe_to_tensor(frame))

    cap.release()
    return loss / count


class Inference:
    def __init__(
        self,
        model_class,
        input_frame_num: int,
        model_path: str,
        video_path: str,
        device: str = "cuda",
        first_frame: Union[int, None] = None,
    ):
        self.model = model_class(input_frame_num).to(device)
        self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
        self.video_path = video_path
        self.input_frame_num = input_frame_num
        self.device = device

        self.cap = cv2.VideoCapture(video_path)

        # Skip the first few frames to match the first_frame
        if first_frame is None or first_frame < input_frame_num:
            first_frame = input_frame_num
        for _ in range(first_frame - input_frame_num):
            _, _ = self.cap.read()

        # Read the few frames to match the input_frame_num
        self.imgs = list()
        for _ in range(input_frame_num):
            _, frame = self.cap.read()
            self.imgs.append(cvframe_to_tensor(frame))

    def __del__(self):
        self.cap.release()

    def __iter__(self):
        # Start the video style transfer
        while True:
            # Pass the input tensor through the model
            with torch.no_grad():
                input_tensor = torch.cat(self.imgs, dim=0).unsqueeze(0).to(self.device)
                *_, output_tensor = self.model(input_tensor)
                output_tensor = output_tensor.clamp(0, 255)

            # Convert output tensor back to image format
            output_image = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            output_image = output_image.astype("uint8")

            yield output_image

            # Read the next frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Update the input tensor list
            self.imgs.pop(0)
            self.imgs.append(cvframe_to_tensor(frame))
