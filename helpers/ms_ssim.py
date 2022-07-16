"""
https://github.com/lizhengwei1992/MS_SSIM_pytorch/blob/master/loss.py
"""

import torch
import torch.nn.functional as F
from math import exp

from torch.nn import L1Loss


def gaussian(window_size, sigma, device):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)],
                         device=device)
    return gauss / gauss.sum()


def create_window(window_size, sigma, channel, device):
    _1D_window = gaussian(window_size, sigma, device).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, size_average=True, channel=3, max_val=1.0, sigma=1.5):
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = sigma * window_size / 11
    window = create_window(window_size, sigma, channel, device=img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    l_map = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    mcs_map = V1 / V2

    # remove padding
    cl = (window_size - 1) // 2
    ssim_map = ssim_map[:, :, cl:-cl, cl:-cl]
    mcs_map = mcs_map[:, :, cl:-cl, cl:-cl]
    l_map = l_map[:, :, cl:-cl, cl:-cl]

    if size_average:
        return ssim_map.mean(), mcs_map.mean()
    return ssim_map, mcs_map, l_map


class MS_SSIM(torch.nn.Module):
    def __init__(self, size_average=True, max_val=255):
        super(MS_SSIM, self).__init__()
        self.size_average = size_average
        self.channel = 3
        self.max_val = max_val

    def ms_ssim(self, img1, img2, levels=5):
        weight = torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device=img1.device)

        msssim = torch.Tensor(levels, ).to(device=img1.device)
        mcs = torch.Tensor(levels, ).to(device=img1.device)
        for i in range(levels):
            ssim_map, mcs_map = ssim(img1, img2, channel=self.channel, max_val=self.max_val)
            msssim[i] = ssim_map
            mcs[i] = mcs_map
            filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
            filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
            img1 = filtered_im1
            img2 = filtered_im2

        value = (torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) *
                 (msssim[levels - 1] ** weight[levels - 1]))
        return value

    def forward(self, img1, img2):
        return self.ms_ssim(img1, img2)


class MixLoss(torch.nn.Module):
    def __init__(self, alpha=0.84):
        super().__init__()
        self.sigmas = [0.5, 1, 2, 4, 8]
        self.window_size = 11
        self.channel = 3
        self.max_val = 1.0
        self.alpha = alpha

        self.l1_loss = L1Loss()

    def forward(self, img1, img2):
        # compute l_0, cs_0
        _, cs_0, l_0 = ssim(img1, img2, channel=self.channel, max_val=self.max_val, size_average=False,
                            sigma=self.sigmas[0])

        result = torch.tensor(1.0, device=img1.device)
        result *= cs_0.mean()

        for i, sigma in enumerate(self.sigmas[1:]):
            window = create_window(self.window_size, sigma, self.channel, device=img1.device)
            cs_i = F.conv2d(cs_0, window, groups=self.channel)

            if i == len(self.sigmas[1:]) - 1:
                l_m = F.conv2d(l_0, window, groups=self.channel)
                cs_i *= l_m

            result *= cs_i.mean()

        # add l1 loss
        return self.alpha * (1 - result) + (1.0 - self.alpha) * self.l1_loss(img1, img2)
