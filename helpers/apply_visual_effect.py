from torch import nn

from effects.xdog import XDoGEffect
from helpers.visual_parameter_def import portrait_preset


class ApplyVisualEffect(nn.Module):
    def __init__(self, effect_type=XDoGEffect,
                 param_names=("blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness"),
                 default_preset=portrait_preset):
        super().__init__()
        self.effect = effect_type()
        self.effect.enable_checkpoints()

        self.param_names = param_names
        self.default_preset = default_preset

    def forward(self, x, visual_parameters):
        default = self.effect.vpd.preset_tensor(self.default_preset, x, True)
        visual_parameters = self.effect.vpd.update_visual_parameters(default, self.param_names,
                                                                     visual_parameters)
        return self.effect(x, visual_parameters)
