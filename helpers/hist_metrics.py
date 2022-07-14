from functools import reduce

import torch
from torch import nn

from helpers.color_conversion import rgb_to_yuv
from helpers.hist_layers import JointHistLayer, SingleDimHistLayer


class EarthMoversDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # input has dims: (Batch x Bins)
        bins = x.size(1)
        r = torch.arange(bins, device=x.device)
        s, t = torch.meshgrid(r, r)
        tt = t >= s

        cdf_x = torch.matmul(x, tt.float())
        cdf_y = torch.matmul(y, tt.float())

        # Note: it feels like we should divide by K here
        # but this is in line with the paper
        return torch.sum(torch.square(cdf_x - cdf_y), dim=1)


class MutualInformationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, p1, p2, p12):
        # input p12 has dims: (Batch x Bins x Bins)
        # input p1 & p2 has dims: (Batch x Bins)

        product_p = torch.matmul(torch.transpose(p1.unsqueeze(1), 1, 2), p2.unsqueeze(1)) + torch.finfo(p1.dtype).eps
        mi = torch.sum(p12 * torch.log(p12 / product_p + torch.finfo(p1.dtype).eps), dim=(1, 2))
        h = -torch.sum(p12 * torch.log(p12 + torch.finfo(p1.dtype).eps), dim=(1, 2))

        return 1 - (mi / h)


class DeepHistLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_hist_layer = JointHistLayer()
        self.single_dim_hist_layer = SingleDimHistLayer()
        self.emd = EarthMoversDistanceLoss()
        self.mi = MutualInformationLoss()

    def forward(self, x, y):
        # x and y are RGB images
        assert x.size(1) == 3
        assert y.size(1) == 3

        # this does not work
        # x = resize_keep_aspect_ratio(x)
        # y = resize_keep_aspect_ratio(y)

        x = rgb_to_yuv(x)
        y = rgb_to_yuv(y)
        
        hist_x = [self.single_dim_hist_layer(x[:, i]) for i in range(3)]
        hist_y = [self.single_dim_hist_layer(y[:, i]) for i in range(3)]
        hist_xy = [self.joint_hist_layer(x[:, i], y[:, i]) for i in range(3)]

        emd_loss = reduce(torch.add, [self.emd(hist_x[i], hist_y[i]) for i in range(3)]) / 3.0
        mi_loss = reduce(torch.add, [self.mi(hist_x[i], hist_y[i], hist_xy[i]) for i in range(3)]) / 3.0

        # factors from paper
        return 100 * emd_loss + 25 * mi_loss
