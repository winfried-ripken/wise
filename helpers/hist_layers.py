from torch import nn, sigmoid
import torch


def phi_k(x, L, W):
    return sigmoid((x + (L / 2)) / W) - sigmoid((x - (L / 2)) / W)


def compute_pj(x, K, L, W):
    mu_k = (L * (torch.arange(K, device=x.device) + 0.5)).view(-1, 1)

    # we assume that x has only one channel already
    # flatten spatial dims
    x = x.reshape(x.size(0), 1, -1)
    x = x.repeat(1, K, 1)  # construct K channels

    # apply activation functions
    return phi_k(x - mu_k, L, W)


class HistLayerBase(nn.Module):
    def __init__(self):
        super().__init__()

        self.K = 256
        self.L = 1 / self.K  # 2 / K -> if values in [-1,1] (Paper)
        self.W = self.L / 2.5


class SingleDimHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        N = x.size(1) * x.size(2)
        pj = compute_pj(x, self.K, self.L, self.W)
        return pj.sum(dim=2) / N


class JointHistLayer(HistLayerBase):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        N = x.size(1) * x.size(2)
        p1 = compute_pj(x, self.K, self.L, self.W)
        p2 = compute_pj(y, self.K, self.L, self.W)
        return torch.matmul(p1, torch.transpose(p2, 1, 2)) / N
