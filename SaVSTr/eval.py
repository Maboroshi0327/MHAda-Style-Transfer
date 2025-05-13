import torch
from torch import nn
from torch.nn import functional as F

import cv2
import numpy as np
import scipy.stats
import argparse

import lpips
from network import VGG19
from utilities import cv2_to_tensor


def lpips_loss(opt, no_print=False):
    # Initializing the model
    if not hasattr(lpips_loss, "loss_fn"):
        lpips_loss.loss_fn = lpips.LPIPS(net="vgg").to(opt.device)
    loss_fn = lpips_loss.loss_fn

    # Load images
    img0 = lpips.im2tensor(lpips.load_image(opt.path0)).to(opt.device)
    img1 = lpips.im2tensor(lpips.load_image(opt.path1)).to(opt.device)

    # Compute distance
    dist01 = loss_fn.forward(img0, img1)

    if not no_print:
        print("Distance: %f" % dist01.item())
    else:
        return dist01.item()


def compute_histogram(img, channel=None):
    # Extract the specified channel, flatten it into a one-dimensional array
    # and use np.bincount to calculate the histogram.
    if channel == None:
        channel_data = img.flatten()
    else:
        channel_data = img[:, :, channel].flatten()
    hist = np.bincount(channel_data, minlength=256) + 1
    return hist


def kl_loss(opt, no_print=False):
    img = cv2.imread(opt.path0)
    s = cv2.imread(opt.path1)

    # Calculate histograms for each channel
    hist_img = [compute_histogram(img, ch) for ch in range(3)]
    hist_s = [compute_histogram(s, ch) for ch in range(3)]

    # Calculate KL divergence for each channel
    KL = 0.0
    for i in range(3):
        KL += scipy.stats.entropy(hist_img[i], hist_s[i])

    KL = KL.item() / 3.0

    if not no_print:
        print("KL: %f" % KL)
    else:
        return KL


def gram_matrix(x: torch.Tensor):
    (b, ch, h, w) = x.size()
    features = x.reshape(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (h * w)
    return gram


def gram_loss(opt, no_print=False):
    img = cv2.imread(opt.path0)
    img = cv2_to_tensor(img)
    img = img.unsqueeze(0).to(opt.device)

    s = cv2.imread(opt.path1)
    s = cv2_to_tensor(s).to(opt.device)
    s = s.unsqueeze(0).to(opt.device)

    if not hasattr(gram_loss, "vgg19"):
        gram_loss.vgg19 = VGG19().to(opt.device)
        gram_loss.vgg19.eval()
    vgg19 = gram_loss.vgg19

    loss = 0.0
    for i in [1, 2, 3, 4, 5]:
        with torch.no_grad():
            fcs = vgg19(img)
            fs = vgg19(s)

        gram_fcs = gram_matrix(fcs[f"relu{i}_1"])
        gram_fs = gram_matrix(fs[f"relu{i}_1"])

        loss += F.mse_loss(gram_fcs, gram_fs)

    loss = loss.item() / 5.0

    if not no_print:
        print("Gram Loss: %f" % loss)
    else:
        return loss


def nth_order_moment(opt, no_print=False):
    img = cv2.imread(opt.path0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = compute_histogram(img)
    hist_p = hist / np.sum(hist)
    hist_mean = np.mean(hist)

    nth_moment = 0.0
    for i in range(256):
        nth_moment += ((hist[i] - hist_mean) ** 2) * hist_p[i]

    if not no_print:
        print("Nth Order Moment: %f" % nth_moment)
    else:
        return nth_moment


def uniformity(opt, no_print=False):
    img = cv2.imread(opt.path0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = compute_histogram(img)
    hist_p = hist / np.sum(hist)

    uniformity = 0.0
    for i in range(256):
        uniformity += hist_p[i] ** 2

    if not no_print:
        print("Uniformity: %f" % uniformity)
    else:
        return uniformity


def average_entropy(opt, no_print=False):
    img = cv2.imread(opt.path0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist = compute_histogram(img)
    hist_p = hist / np.sum(hist)

    entropy = 0.0
    for i in range(256):
        if hist_p[i] > 0:
            entropy -= hist_p[i] * np.log2(hist_p[i])

    if not no_print:
        print("Average Entropy: %f" % entropy)
    else:
        return entropy


class SSIMMetric(nn.Module):
    def __init__(self, window_size: int = 11, channel: int = 3, sigma: float = 1.5, reduction: str = "mean"):
        """
        window_size: 高斯核大小，通常 11
        channel: 圖片通道數(RGB=3, 灰階=1)
        sigma: 高斯核的標準差
        reduction: 'mean' 或 'none'，控制最終輸出的聚合方式
        """
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.reduction = reduction
        # 建立並註冊高斯核（不參與梯度計算）
        _1D = torch.linspace(-(window_size // 2), window_size // 2, window_size)
        gauss = torch.exp(-(_1D**2) / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        _2D = gauss[:, None] @ gauss[None, :]
        kernel = _2D.expand(channel, 1, window_size, window_size).contiguous()
        self.register_buffer("kernel", kernel)

        # 穩定用常數
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        img1, img2: Tensor[B, C, H, W]，值域建議歸一化到 [0,1]
        return: SSIM 分數，若 reduction='mean'，則回傳標量；否則回傳 [B] 的向量
        """
        assert img1.shape == img2.shape and img1.dim() == 4, "輸入需為 [B,C,H,W] 且相同尺寸"

        # 計算局部均值
        mu1 = F.conv2d(img1, self.kernel, padding=self.window_size // 2, groups=self.channel)
        mu2 = F.conv2d(img2, self.kernel, padding=self.window_size // 2, groups=self.channel)

        # 計算局部方差與協方差
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, self.kernel, padding=self.window_size // 2, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, self.kernel, padding=self.window_size // 2, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, self.kernel, padding=self.window_size // 2, groups=self.channel) - mu1_mu2

        # SSIM 公式
        num = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        den = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = num / den  # [B, C, H, W]

        # 平均通道與空間
        ssim_per_channel = ssim_map.mean(dim=[2, 3])  # [B, C]
        ssim_per_image = ssim_per_channel.mean(dim=1)  # [B]

        if self.reduction == "mean":
            return ssim_per_image.mean()
        else:
            return ssim_per_image


def ssim_loss(opt, no_print=False):
    img = cv2.imread(opt.path0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2_to_tensor(img).unsqueeze(0).to(opt.device)

    s = cv2.imread(opt.path1)
    s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
    s = cv2_to_tensor(s).unsqueeze(0).to(opt.device)

    if not hasattr(ssim_loss, "metric"):
        ssim_loss.metric = SSIMMetric(window_size=11, channel=3, sigma=1.5, reduction="mean").to(opt.device)
    ssim_metric = ssim_loss.metric
    ssim = ssim_metric(img, s)

    if not no_print:
        print("SSIM: %f" % ssim.item())
    else:
        return ssim.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage="eval.py [-h] [-m MODE] [-p0 PATH0] [-p1 PATH1] [-d DEVICE]",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=35, width=120),
    )
    parser.add_argument("-m", "--mode", type=str, default="lpips", help="mode of the evaluation, default is lpips")
    parser.add_argument("-p0", "--path0", type=str, default="./results/stylized.png", help="path to the stylized image")
    parser.add_argument("-p1", "--path1", type=str, default="./results/style.png", help="path to the content/style image")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device to use, default is cuda")
    opt = parser.parse_args()

    if opt.mode == "lpips":
        lpips_loss(opt)
    elif opt.mode == "ssim":
        ssim_loss(opt)
    elif opt.mode == "kl":
        kl_loss(opt)
    elif opt.mode == "gram":
        gram_loss(opt)
    elif opt.mode == "moment":
        nth_order_moment(opt)
    elif opt.mode == "uni":
        uniformity(opt)
    elif opt.mode == "entropy":
        average_entropy(opt)
