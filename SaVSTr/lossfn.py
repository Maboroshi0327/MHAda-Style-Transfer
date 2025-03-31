import torch
from torch import linalg as LA


def style_loss(fcs, fs, loss_fn):
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


def content_loss(fcs, fc, loss_fn):
    loss = 0
    for i in [4, 5]:
        loss += loss_fn(fcs[f"relu{i}_1"], fc[f"relu{i}_1"])

    return loss


def identity_loss_1(cc, c, ss, s, loss_fn):
    return loss_fn(cc, c) + loss_fn(ss, s)


def identity_loss_2(fcc, fc, fss, fs, loss_fn):
    loss = 0
    for i in [1, 2, 3, 4, 5]:
        loss += loss_fn(fcc[f"relu{i}_1"], fc[f"relu{i}_1"])
        loss += loss_fn(fss[f"relu{i}_1"], fs[f"relu{i}_1"])

    return loss
