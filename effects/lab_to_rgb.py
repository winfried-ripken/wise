import torch

from helpers import color_conversion


class LabToRGBEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.clamp(color_conversion.lab_to_rgb(torch.clamp(x, 0.0, 1.0)), 0.0, 1.0)
