import torch

from helpers.index_helper import IndexHelper


class EdgeBlendEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.edgecolor = [0.0, 0.0, 0.0]

    def forward(self, x, edge, edge_alpha=1.0):
        x_rgb, x_a = IndexHelper.get_rgb_and_alpha(x)
        edgecolor = torch.tensor(self.edgecolor, device=x.device).reshape(1, -1, 1, 1)
        edge = edge * edge_alpha
        res = torch.lerp(x_rgb, edgecolor, edge)
        if x_a is not None:
            res = IndexHelper.cat(res, x_a)
        return res
