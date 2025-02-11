import os
from typing import Union

import cv2
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from flowlib import read
from utilities import list_files, visualize_flow, warp, flow_warp_mask, toTensor255, toTensor, toPil


class Coco2014(Dataset):
    def __init__(self, path: str, resolution: tuple = (256, 256)):
        """
        path -> Path to the location where the "coco2014" folder is kept. \\
        resolution -> Resolution of the images to be returned. Width first, then height.
        """
        super().__init__()
        self.path = os.path.join(path, "train2014")
        self.resolution = resolution

        self.paths = list()
        files = list_files(self.path)
        for file in files:
            self.paths.append(file)

        self.length = len(self.paths)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB").resize(self.resolution, Image.BILINEAR)
        img = toTensor255(img)
        return img


class FlyingThings3D(Dataset):
    def __init__(self, path: str, resolution: tuple = (640, 360), frame_num: int = 1):
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

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, path: str, resolution: tuple = (640, 360), frame_num: int = 1):
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

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    def __init__(self, path: Union[str, list], resolution: tuple = (640, 360), frame_num: int = 1):
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

    def __getitem__(self, idx: int) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < len(self.monkaa):
            return self.monkaa[idx]
        else:
            return self.flyingthings3d[idx - len(self.monkaa)]


def test_FlyingThings3D_Monkaa(frame_num: int = 4):
    data = FlyingThings3D_Monkaa(["C:\\Datasets\\monkaa", "D:\\Datasets\\flyingthings3d"], frame_num=frame_num)

    pbar = tqdm(range(10), desc="Test FlyingThings3D_Monkaa", leave=True)
    for i in pbar:
        img1, img2, flow_into_past, mask = data[i * 2000]

        # get index
        index = (frame_num - 1) * 3
        index = [i for i in range(index, index + 3)]

        # warp image & visualize flow
        next_img = warp(img1[index].unsqueeze(0), flow_into_past.unsqueeze(0)).squeeze(0)
        warp_mask = mask * next_img
        flow_rgb = visualize_flow(flow_into_past)

        # convert to PIL
        img1 = toPil(img1[index].byte())
        img2 = toPil(img2[index].byte())
        next_img = toPil(next_img.byte())
        warp_mask = toPil(warp_mask.byte())
        mask = toPil(mask)

        # create directory if it doesn't exist
        save_dir = f"./datasets_images/{i + 1}/"
        os.makedirs(save_dir, exist_ok=True)

        # delete old images
        files = list_files(save_dir)
        for f in files:
            os.remove(f)

        # save images
        img1.save(os.path.join(save_dir, "img1.png"))
        img2.save(os.path.join(save_dir, "img2.png"))
        # mask.save(os.path.join(save_dir, "mask.png"))
        next_img.save(os.path.join(save_dir, "next_img.png"))
        warp_mask.save(os.path.join(save_dir, "warp_mask.png"))
        cv2.imwrite(os.path.join(save_dir, "flow.png"), flow_rgb)


if __name__ == "__main__":
    test_FlyingThings3D_Monkaa()
