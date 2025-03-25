import torch
from torch import linalg as LA


def global_stylized_loss(fcs, fs, loss_fn):
    # Mean distance
    fcs_mean = fcs.mean(dim=(2, 3))
    fs_mean = fs.mean(dim=(2, 3))
    mean_dist = loss_fn(fcs_mean, fs_mean)

    # Standard deviation distance
    fcs_std = fcs.std(dim=(2, 3))
    fs_std = fs.std(dim=(2, 3))
    std_dist = loss_fn(fcs_std, fs_std)

    # Loss for each ReLU_x_1 layer
    return mean_dist + std_dist


def local_feature_loss(fcs, fc, loss_fn):
    dist = loss_fn(fcs, fc)
    return dist
