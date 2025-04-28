import torch
import torch.nn as nn
from torch.nn import functional as F

from utilities import feature_down_sample, warp


def global_style_loss(fcs, fs, loss_fn):
    loss = 0
    for i in [1, 2, 3, 4, 5]:
        # Mean distance
        fcs_mean = fcs[f"relu{i}_1"].mean(dim=(2, 3))
        fs_mean = fs[f"relu{i}_1"].mean(dim=(2, 3))
        mean_dist = loss_fn(fcs_mean, fs_mean)

        # Standard deviation distance
        fcs_std = fcs[f"relu{i}_1"].std(dim=(2, 3))
        fs_std = fs[f"relu{i}_1"].std(dim=(2, 3))
        std_dist = loss_fn(fcs_std, fs_std)

        # Loss for each ReLU_x_1 layer
        loss += mean_dist + std_dist

    return loss


def local_feature_loss(fc, fs, fcs, adaattnNoLearn, loss_fn):
    loss = 0
    for idx, i in enumerate([3, 4, 5]):
        c_1x = feature_down_sample(fc, i)
        s_1x = feature_down_sample(fs, i)
        adaattn = adaattnNoLearn[idx](fc[f"relu{i}_1"], fs[f"relu{i}_1"], c_1x, s_1x)
        loss += loss_fn(fcs[f"relu{i}_1"], adaattn)

    return loss


def identity_loss_1(cc, c, ss, s, loss_fn):
    return loss_fn(cc, c) + loss_fn(ss, s)


def identity_loss_2(fcc, fc, fss, fs, loss_fn):
    loss = 0
    for i in [1, 2, 3, 4, 5]:
        loss += loss_fn(fcc[f"relu{i}_1"], fc[f"relu{i}_1"])
        loss += loss_fn(fss[f"relu{i}_1"], fs[f"relu{i}_1"])

    return loss


def temporal_loss(cs1, cs2, flow, mask, loss_fn):
    mask = mask.unsqueeze(1)
    mask = mask.expand(-1, cs1.shape[1], -1, -1)
    non_zero_count = mask.sum() + 1e-8
    warped_style = warp(cs1, flow)
    loss = mask * loss_fn(cs2, warped_style)
    loss = loss.sum() / non_zero_count
    return loss
