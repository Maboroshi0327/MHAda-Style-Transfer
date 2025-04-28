import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import os
import random
from PIL import Image
from tqdm import tqdm
from typing import Tuple, Union

from flowlib import read
from utilities import toTensor255, toTensor, toPil, toTensorCrop, list_files, flow_warp_mask


def coco(path="../datasets/coco", size_crop: tuple = (256, 256)):
    """
    size_crop: (height, width)
    """
    dataset = ImageFolder(root=path, transform=toTensorCrop(size_crop=size_crop))
    return dataset


def wikiArt(path="../datasets/WikiArt", size_crop: tuple = (256, 256)):
    """
    size_crop: (height, width)
    """
    dataset = ImageFolder(root=path, transform=toTensorCrop(size_crop=size_crop))
    return dataset


class WikiArt(Dataset):
    def __init__(self, image_size: tuple = (256, 256), path="../datasets/WikiArt", length=None):
        self.wikiart = wikiArt(path, image_size)
        self.wikiart_len = len(self.wikiart)
        if length is not None:
            self.length = length
        else:
            self.length = self.wikiart_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        wikiart_idx = random.randint(0, self.wikiart_len - 1)
        return self.wikiart[wikiart_idx][0]


class CocoWikiArt(Dataset):
    def __init__(self, image_size: tuple = (256, 256), coco_path="../datasets/coco", wikiart_path="../datasets/WikiArt"):
        self.coco = coco(coco_path, image_size)
        self.wikiart = wikiArt(wikiart_path, image_size)
        self.coco_len = len(self.coco)
        self.wikiart_len = len(self.wikiart)

    def __len__(self):
        return self.coco_len

    def __getitem__(self, idx):
        wikiart_idx = random.randint(0, self.wikiart_len - 1)
        return self.coco[idx][0], self.wikiart[wikiart_idx][0]


class ImageNet1k(Dataset):
    def __init__(self, image_size: tuple = (256, 256), path="../datasets/ImageNet1K", mode: str = "train"):
        path = os.path.join(path, mode)

        if mode == "train":
            self.dataset = ImageFolder(root=path, transform=toTensorCrop(size_resize=(300, 300), size_crop=image_size))
        elif mode == "val":
            self.dataset = ImageFolder(root=path, transform=toTensorCrop(size_resize=(256, 256), size_crop=(256, 256)))
        else:
            raise ValueError("Mode must be 'train' or 'val'.")

        self.image_size = image_size
        self.path = path
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        label = self.dataset[idx][1]
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=1000).float()
        return self.dataset[idx][0], one_hot_label


class FlyingThings3D(Dataset):
    def __init__(self, path: str, resolution: tuple = (512, 256), frame_num: int = 1):
        """
        path -> Path to the location where the "frames_finalpass", "optical_flow" and "motion_boundaries" folders are kept inside the FlyingThings3D folder. \\
        resolution -> Resolution of the images to be returned. Width first, then height. \\
        frame_num -> Number of frames to be returned. Must be between 1 and 9.
        """
        super().__init__()
        path_frame = os.path.join(path, "frames_finalpass/TRAIN")
        path_flow = os.path.join(path, "optical_flow/TRAIN")
        path_motion = os.path.join(path, "motion_boundaries/TRAIN")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert os.path.exists(path_flow), f"Path {path_flow} does not exist."
        assert os.path.exists(path_motion), f"Path {path_motion} does not exist."
        assert len(resolution) == 2 and isinstance(resolution, tuple), "Resolution must be a tuple of 2 integers."
        assert 1 <= frame_num and frame_num <= 9, "Frame number must be between 1 and 9."

        self.path = path
        self.frame = list()
        self.flow = list()
        self.motion = list()

        # progress bar
        pbar = tqdm(desc="Initial FlyingThings3D", total=2239 * (10 - frame_num) * 3)

        # frames_finalpass
        for abcpath in ["A", "B", "C"]:
            for folder in os.listdir(os.path.join(path_frame, abcpath)):
                files = list_files(os.path.join(path_frame, abcpath, folder, "left"))
                for i in range(10 - frame_num):
                    self.frame.append(files[i : i + frame_num + 1])
                    pbar.update(1)

        # optical_flow
        for abcpath in ["A", "B", "C"]:
            for folder in os.listdir(os.path.join(path_flow, abcpath)):
                files_into_future = list_files(os.path.join(path_flow, abcpath, folder, "into_future", "left"))
                files_into_past = list_files(os.path.join(path_flow, abcpath, folder, "into_past", "left"))
                for i in range(10 - frame_num):
                    self.flow.append((files_into_future[i + frame_num - 1], files_into_past[i + frame_num]))
                    pbar.update(1)

        # mask
        for abcpath in ["A", "B", "C"]:
            for folder in os.listdir(os.path.join(path_motion, abcpath)):
                files = list_files(os.path.join(path_motion, abcpath, folder, "into_future", "left"))
                for i in range(10 - frame_num):
                    self.motion.append(files[i + frame_num])
                    pbar.update(1)

        self.length = len(self.frame)
        self.resolution = resolution
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        idx -> Index of the image pair, optical flow and mask to be returned.
        """
        # read image
        imgs = list()
        for path in self.frame[idx]:
            img = Image.open(path).convert("RGB").resize(self.resolution, Image.BILINEAR)
            img = toTensor255(img)
            imgs.append(img)
        img1 = torch.cat(imgs[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs[1 : self.frame_num + 1], dim=0)

        # read flow
        flow_into_future = toTensor(read(self.flow[idx][0]).copy())[:-1]
        flow_into_past = toTensor(read(self.flow[idx][1]).copy())[:-1]
        originalflowshape = flow_into_past.shape

        flow_into_past = F.interpolate(
            flow_into_past.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        flow_into_future = F.interpolate(
            flow_into_future.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        flow_into_future[0] *= flow_into_future.shape[1] / originalflowshape[1]
        flow_into_future[1] *= flow_into_future.shape[2] / originalflowshape[2]
        flow_into_past[0] *= flow_into_past.shape[1] / originalflowshape[1]
        flow_into_past[1] *= flow_into_past.shape[2] / originalflowshape[2]

        # read motion boundaries
        motion = Image.open(self.motion[idx]).resize(self.resolution, Image.BILINEAR)
        motion = toTensor(motion).squeeze(0)
        motion[motion != 0] = 1
        motion = 1 - motion

        # create mask
        mask = flow_warp_mask(flow_into_future, flow_into_past)
        mask = mask * motion

        return img1, img2, flow_into_past, mask


class Monkaa(Dataset):
    def __init__(self, path: str, resolution: tuple = (512, 256), frame_num: int = 1):
        """
        path -> Path to the location where the "frames_finalpass", "optical_flow" and "motion_boundaries" folders are kept inside the Monkaa folder. \\
        resolution -> Resolution of the images to be returned. Width first, then height. \\
        frame_num -> Number of frames to be returned. Must be between 1 and 9.
        """
        super().__init__()
        path_frame = os.path.join(path, "frames_finalpass")
        path_flow = os.path.join(path, "optical_flow")
        path_motion = os.path.join(path, "motion_boundaries")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert os.path.exists(path_flow), f"Path {path_flow} does not exist."
        assert os.path.exists(path_motion), f"Path {path_motion} does not exist."
        assert len(resolution) == 2 and isinstance(resolution, tuple), "Resolution must be a tuple of 2 integers."
        assert 1 <= frame_num and frame_num <= 9, "Frame number must be between 1 and 9."

        self.path = path
        self.frame = list()
        self.flow = list()
        self.motion = list()

        # count number of data
        data_num = 0
        for folder in os.listdir(path_frame):
            files = list_files(os.path.join(path_frame, folder, "left"))
            data_num += len(files) - frame_num

        # progress bar
        pbar = tqdm(desc="Initial Monkaa", total=data_num * 3)

        for folder in os.listdir(path_frame):
            files = list_files(os.path.join(path_frame, folder, "left"))
            for i in range(len(files) - frame_num):
                self.frame.append(files[i : i + frame_num + 1])
                pbar.update(1)

        for folder in os.listdir(path_flow):
            files_into_future = list_files(os.path.join(path_flow, folder, "into_future", "left"))
            files_into_past = list_files(os.path.join(path_flow, folder, "into_past", "left"))
            for i in range(len(files_into_future) - frame_num):
                self.flow.append((files_into_future[i + frame_num - 1], files_into_past[i + frame_num]))
                pbar.update(1)

        for folder in os.listdir(path_motion):
            files = list_files(os.path.join(path_motion, folder, "into_future", "left"))
            for i in range(len(files) - frame_num):
                self.motion.append(files[i + frame_num])
                pbar.update(1)

        self.length = len(self.frame)
        self.resolution = resolution
        self.frame_num = frame_num

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        idx -> Index of the image pair, optical flow and mask to be returned.
        """
        # read image
        imgs = list()
        for path in self.frame[idx]:
            img = Image.open(path).convert("RGB").resize(self.resolution, Image.BILINEAR)
            img = toTensor255(img)
            imgs.append(img)
        img1 = torch.cat(imgs[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs[1 : self.frame_num + 1], dim=0)

        # read flow
        flow_into_future = toTensor(read(self.flow[idx][0]).copy())[:-1]
        flow_into_past = toTensor(read(self.flow[idx][1]).copy())[:-1]
        originalflowshape = flow_into_past.shape

        flow_into_past = F.interpolate(
            flow_into_past.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        flow_into_future = F.interpolate(
            flow_into_future.unsqueeze(0),
            size=(self.resolution[1], self.resolution[0]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        flow_into_future[0] *= flow_into_future.shape[1] / originalflowshape[1]
        flow_into_future[1] *= flow_into_future.shape[2] / originalflowshape[2]
        flow_into_past[0] *= flow_into_past.shape[1] / originalflowshape[1]
        flow_into_past[1] *= flow_into_past.shape[2] / originalflowshape[2]

        # read motion boundaries
        motion = Image.open(self.motion[idx]).resize(self.resolution, Image.BILINEAR)
        motion = toTensor(motion).squeeze(0)
        motion[motion != 0] = 1
        motion = 1 - motion

        # create mask
        mask = flow_warp_mask(flow_into_future, flow_into_past)
        mask = mask * motion

        return img1, img2, flow_into_past, mask


class FlyingThings3D_Monkaa(Dataset):
    def __init__(self, path: Union[str, list], resolution: tuple = (512, 256), frame_num: int = 1):
        """
        path -> Path to the location where the "monkaa" and "flyingthings3d" folders are kept.
                If path is a list, then the first element is the path to the "monkaa" folder and the second element is the path to the "flyingthings3d" folder.
        resolution -> Resolution of the images to be returned. Width first, then height.
        """
        super().__init__()

        if isinstance(path, str):
            self.monkaa = Monkaa(os.path.join(path, "monkaa"), resolution, frame_num)
            self.flyingthings3d = FlyingThings3D(os.path.join(path, "flyingthings3d"), resolution, frame_num)
        elif isinstance(path, list):
            self.monkaa = Monkaa(path[0], resolution, frame_num)
            self.flyingthings3d = FlyingThings3D(path[1], resolution, frame_num)
        else:  # pragma: no cover
            raise ValueError("Path must be a string or a list of strings.")

        self.length = len(self.monkaa) + len(self.flyingthings3d)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < len(self.monkaa):
            return self.monkaa[idx]
        else:
            return self.flyingthings3d[idx - len(self.monkaa)]


class FlyingThings3D_Monkaa_Coco_WikiArt(Dataset):
    def __init__(self, image_size1: tuple = (256, 256), image_size2: tuple = (256, 512), path="../datasets"):
        self.coco = coco(os.path.join(path, "coco"), image_size1)
        self.wikiart = wikiArt(os.path.join(path, "WikiArt"), image_size1)
        self.flyingthings3d_monkaa = FlyingThings3D_Monkaa(os.path.join(path, "SceneFlowDatasets"), resolution=(image_size2[1], image_size2[0]))
        self.coco_len = len(self.coco)
        self.wikiart_len = len(self.wikiart)
        self.flyingthings3d_monkaa_len = len(self.flyingthings3d_monkaa)

        self.length = self.flyingthings3d_monkaa_len

    def __len__(self):
        return self.length

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, ...]]:
        coco_idx = random.randint(0, self.coco_len - 1)
        wikiart_idx = random.randint(0, self.wikiart_len - 1)

        return (
            self.coco[coco_idx][0],
            self.wikiart[wikiart_idx][0],
            *self.flyingthings3d_monkaa[idx],
        )


if __name__ == "__main__":
    dataset = CocoWikiArt()
    c, s = dataset[123]
    print("CocoWikiArt dataset")
    print("dataset length:", len(dataset))

    from utilities import toPil

    toPil(c.byte()).save("coco.png")
    toPil(s.byte()).save("wikiart.png")
    print("Saved coco.png and wikiart.png")
