import torch

from helpers.index_helper import IndexHelper


class OilpaintComposeEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, color, bump_map):
        i = IndexHelper(color)
        color_rgb, alpha = i.get_rgb_and_alpha(color)
        bump_clamped = torch.clamp(color_rgb * i.idx(bump_map, "x") + i.idx(bump_map, 'y'), min=0.0, max=1.0)
        final_color = torch.lerp(color_rgb, bump_clamped, alpha)
        return final_color
