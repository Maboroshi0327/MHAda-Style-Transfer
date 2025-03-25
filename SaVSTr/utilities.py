import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import os
from typing import Union

import cv2


toTensor255 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
    ]
)
toTensor = transforms.ToTensor()
toPil = transforms.ToPILImage()
gaussianBlur = transforms.GaussianBlur(kernel_size=3, sigma=1.0)


def toTensorCrop(size_resize: tuple = (512, 512), size_crop: tuple = (256, 256)):
    """
    size_resize: (height, width) \\
    size_crop: (height, width)
    """
    transform = transforms.Compose(
        [
            transforms.Resize(size_resize),
            transforms.RandomCrop(size=size_crop),
            toTensor255,
        ]
    )
    return transform


def cvframe_to_tensor(frame, resize: Union[tuple, None] = None):
    """
    resize: (width, height)
    """
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if resize is not None:
        frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)

    return toTensor255(frame)


def list_files(directory):
    files = [f.path for f in os.scandir(directory) if f.is_file()]
    return sorted(files)


def list_folders(directory):
    folders = [f.path for f in os.scandir(directory) if f.is_dir()]
    return sorted(folders)


def mkdir(directory, delete_existing_files=False):
    # create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # delete old data
    if delete_existing_files:
        files = list_files(directory)
        for f in files:
            os.remove(f)


def imageNet1k_normalize(batch: torch.Tensor):
    # normalize using imagenet mean and std
    batch = batch.float()
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(batch.device)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(batch.device)
    normalized_batch = (batch / 255.0 - mean) / std
    return normalized_batch


def print_parameters(model):
    for name, _ in model.named_parameters():
        print(name)


def print_state_dict(state_dict):
    for name, _ in state_dict.items():
        print(name)


def feature_down_sample(feat, last_feat_idx):
    size = feat[last_feat_idx].shape[-2:]

    # Downsample the features
    result = list()
    for i in range(last_feat_idx):
        down = F.interpolate(feat[i], size=size, mode="bilinear", align_corners=False)
        result.append(down)
    result.append(feat[last_feat_idx])

    # Concatenate the features
    return torch.cat(result, dim=1)
