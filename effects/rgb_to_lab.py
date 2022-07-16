import torch

from helpers import color_conversion
from helpers.index_helper import IndexHelper


class RGBToLabEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        i = IndexHelper(x)
        x, alpha = i.get_rgb_and_alpha(x)

        return i.combine_rgb_and_alpha(color_conversion.rgb_to_lab(x), alpha)
