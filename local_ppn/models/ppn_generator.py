import torch
from segmentation_models_pytorch import Unet
from torch import nn

from effects import xdog_params
from helpers.apply_visual_effect import ApplyVisualEffect


class OurPPNGenerator(nn.Module):
    def __init__(self, unet_architecture, conv_net, effect, **kwargs):
        super().__init__()
        self.convnet_g = conv_net
        param_names = xdog_params

        if unet_architecture == "classic":
            self.unet = Unet(classes=len(param_names),
                             encoder_weights="imagenet", activation="identity",
                             encoder_name="resnet50")
        elif unet_architecture == "random":
            self.unet = Unet(classes=len(param_names),
                             encoder_weights=None, activation="identity",
                             encoder_name="resnet50")
        elif unet_architecture == "none":
            self.unet = None
            param_names = []  # use default preset
        else:
            raise ValueError("architecture found")

        if effect == "xdog":
            self.apply_visual_effect = ApplyVisualEffect(param_names=param_names)
        else:
            raise ValueError("effect not found")

    def forward(self, x):
        x = self.ppn_part_forward(x)
        return self.conv_part_forward(x)

    def conv_part_forward(self, x):
        return self.convnet_g(x)

    def ppn_part_forward(self, x):
        predicted_param = self.predict_parameters(x)

        x = (x * 0.5) + 0.5
        x = self.apply_visual_effect(x, predicted_param)
        return (x - 0.5) / 0.5

    def predict_parameters(self, x):
        if self.unet is None:
            return None

        parameter_prediction = self.unet(x)
        return torch.tanh(parameter_prediction) * 0.5
