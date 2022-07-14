import torch

from helpers import color_conversion
from helpers.index_helper import IndexHelper


class ColorAdjustmentToonEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, brightness, contrast, saturation, gamma, expose, hueshift):
        i = IndexHelper(x)
        gamma = i.view(gamma)
        gamma = torch.clamp(gamma, min=0.3)

        x = x + i.view(brightness)
        x = x * expose

        x = (x - 0.5) * i.view(contrast) + 0.5
        x = torch.clamp(x, 0.0, 1.0)

        hsv = color_conversion.rgb_to_hsv(x)

        i.set_idx(hsv, "g", torch.clamp(i.idx(hsv, "g") * i.view(saturation), 0.0, 1.0))
        i.set_idx(hsv, "r", torch.frac(i.idx(hsv, "r") + hueshift))

        rgb = color_conversion.hsv_to_rgb(hsv)
        return torch.clamp(i.safe_pow(rgb, 1.0 / gamma), 0.0, 1.0)
