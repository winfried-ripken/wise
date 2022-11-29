from copy import deepcopy

import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

from effects.identity import IdentityEffect
from helpers import np_to_torch
from helpers.index_helper import IndexHelper
from helpers.visual_parameter_def import VisualParameterDef


class EffectBase(torch.nn.Module):
    def __init__(self, vp_ranges):
        super().__init__()
        self.create_checkpoints = False
        self.vpd = VisualParameterDef(deepcopy(vp_ranges))

        self.enable_adapt_hue_preprocess = False
        self.enable_adapt_hue_postprocess = False

    def enable_checkpoints(self):
        self.create_checkpoints = True
        return self

    def disable_checkpoints(self):
        self.create_checkpoints = False
        return self

    def run(self, submodule, *args):
        # do not checkpoint identity
        if self.create_checkpoints and not isinstance(submodule, IdentityEffect):
            return checkpoint(submodule, *args)
        else:
            return submodule(*args)

    def forward_vps(self, vps):
        return self.vpd.scale_parameters(vps)

    def forward(self, x, visual_parameters):
        visual_parameters = self.forward_vps(visual_parameters)
        x = self.forward_effect(x, visual_parameters)
        return IndexHelper.generate_result(x)

    def forward_effect(self, x, visual_parameters):
        raise NotImplementedError('Method needs to be implemented by effect.')

    def load_texture(self, name):
        tex = Image.open(self.tex_path / f"{name}.png").convert("RGB")
        return np_to_torch(tex)
