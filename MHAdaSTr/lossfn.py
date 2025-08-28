import torch
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


def output_level_temporal_loss(c1, c2, cs1, cs2, flow, mask, lossMatrix):
    warped_c1 = warp(c1, flow)
    warped_cs1 = warp(cs1, flow)

    input_term = c2 - warped_c1
    input_term = 0.2126 * input_term[:, 0] + 0.7152 * input_term[:, 1] + 0.0722 * input_term[:, 2]
    input_term = input_term.unsqueeze(1).expand(-1, c2.shape[1], -1, -1)

    output_term = cs2 - warped_cs1

    mask = mask.unsqueeze(1)
    mask = mask.expand(-1, c2.shape[1], -1, -1)

    loss = torch.sum(mask * (lossMatrix(output_term, input_term)))
    non_zero_count = torch.nonzero(mask).shape[0]
    loss *= 1 / non_zero_count
    return loss


def feature_level_temporal_loss(f1, f2, flow, mask, lossMatrix):
    # Warp feature maps
    feature_flow = F.interpolate(flow, size=f1.shape[2:], mode="bilinear")
    feature_flow[:, 0] *= float(f1.shape[3]) / flow.shape[3]
    feature_flow[:, 1] *= float(f1.shape[2]) / flow.shape[2]
    warped_f1 = warp(f1, feature_flow)

    # Create feature mask
    feature_mask = F.interpolate(mask.unsqueeze(1), size=f1.shape[2:], mode="bilinear").squeeze(1)
    feature_mask = (feature_mask > 0).float()
    feature_mask = feature_mask.unsqueeze(1)
    feature_mask = feature_mask.expand(-1, f1.shape[1], -1, -1)

    # Feature-Map-Level Temporal Loss
    loss = torch.sum(feature_mask * lossMatrix(f2, warped_f1))
    non_zero_count = torch.nonzero(feature_mask).shape[0]
    loss *= 1 / non_zero_count
    return loss
