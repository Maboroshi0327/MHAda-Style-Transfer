import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

import os
import random

import cv2
from tqdm import tqdm

from utilities import toTensor255, toPil, toTensorCrop, list_files, list_folders, mkdir


def Coco(path="../datasets/coco", size_crop: tuple = (256, 256)):
    """
    size_crop: (height, width)
    """
    dataset = ImageFolder(root=path, transform=toTensorCrop(size_crop=size_crop))
    return dataset


def WikiArt(path="../datasets/WikiArt", size_crop: tuple = (256, 256)):
    """
    size_crop: (height, width)
    """
    dataset = ImageFolder(root=path, transform=toTensorCrop(size_crop=size_crop))
    return dataset


class CocoWikiArt(Dataset):
    def __init__(self, coco_path="../datasets/coco", wikiart_path="../datasets/WikiArt"):
        self.coco = Coco(coco_path)
        self.wikiart = WikiArt(wikiart_path)
        self.coco_len = len(self.coco)
        self.wikiart_len = len(self.wikiart)

    def __len__(self):
        return self.coco_len

    def __getitem__(self, idx):
        wikiart_idx = random.randint(0, self.wikiart_len - 1)
        return self.coco[idx][0], self.wikiart[wikiart_idx][0]


def get_frames(video_path="../datasets/Videvo", img_size=(512, 256)):
    files = list_files(video_path)

    # progress bar
    pbar = tqdm(desc="Extracting frames", total=len(files))

    videos_idx = 0
    for file in files:
        # create directory if it doesn't exist
        save_dir = f"./Videvo/frames/{videos_idx:05d}/"
        mkdir(save_dir)

        # read video and save frames
        cap = cv2.VideoCapture(file)
        frames_idx = 0
        while True:
            # read video frame
            ret, frame = cap.read()
            if not ret:
                break

            # resize frame and save it
            frame = cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(save_dir, f"{frames_idx:05d}.jpg"), frame)
            frames_idx += 1

        cap.release()
        videos_idx += 1

        pbar.update(1)


class Videvo(Dataset):
    def __init__(self, path: str = "./Videvo", frame_num: int = 1):
        super().__init__()
        path_frame = os.path.join(path, "frames")

        assert os.path.exists(path_frame), f"Path {path_frame} does not exist."
        assert 1 <= frame_num, "Frame number must be equal or greater than 1."

        self.frames = list()
        for folder in list_folders(path_frame):
            files = list_files(folder)
            for i in range(len(files) - frame_num):
                self.frames.append(files[i : i + frame_num + 1])

        self.frame_num = frame_num
        self.length = len(self.frames)
        print(f"Videvo total data: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        # read image
        imgs_gray = list()
        imgs_tensor = list()
        for path in self.frames[idx]:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            imgs_gray.append(img_gray)
            imgs_tensor.append(toTensor255(img))

        img1 = torch.cat(imgs_tensor[0 : self.frame_num], dim=0)
        img2 = torch.cat(imgs_tensor[1 : self.frame_num + 1], dim=0)
        return img1, img2


class VidevoWikiArt(Dataset):
    def __init__(self, videvo_path="./Videvo", wikiart_path="../datasets/WikiArt"):
        self.videvo = Videvo(videvo_path)
        self.wikiart = WikiArt(wikiart_path, size_crop=(256, 512))
        self.videvo_len = len(self.videvo)
        self.wikiart_len = len(self.wikiart)

    def __len__(self):
        return self.videvo_len

    def __getitem__(self, idx):
        wikiart_idx = random.randint(0, self.wikiart_len - 1)
        return self.videvo[idx][0], self.videvo[idx][1], self.wikiart[wikiart_idx][0]


if __name__ == "__main__":
    dataset = CocoWikiArt()
    c, s = dataset[123]
    print("CocoWikiArt dataset")
    print("dataset length:", len(dataset))

    from utilities import toPil
    toPil(c.byte()).save("coco.png")
    toPil(s.byte()).save("wikiart.png")
    print("Saved coco.png and wikiart.png")

    # get_frames()
    # dataset = VidevoWikiArt()
    # c1, c2, s = dataset[10000]
    # toPil(c1.byte()).save("c1.png")
    # toPil(c2.byte()).save("c2.png")
    # toPil(s.byte()).save("s.png")
