import torch

from helpers.index_helper import IndexHelper


class TangentFlowEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        i = IndexHelper(x)

        g_x = x[:, 0:1]
        g_y = x[:, 1:2]
        g_z = x[:, 2:3]

        b = -2.0 * g_z
        a = g_y - g_x

        phi = 0.5 * torch.atan2(b + torch.finfo(b.dtype).eps, a)
        tangent = torch.cat((torch.cos(phi), torch.sin(phi)), dim=1)
        tangent_undefined = (a == 0).logical_and(b == 0)
        clean_tangent = torch.where(tangent_undefined, i.vec2_const(0.0, 0.01), tangent)

        return clean_tangent
