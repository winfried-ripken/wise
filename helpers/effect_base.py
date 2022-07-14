import torch
from torch.utils.checkpoint import checkpoint

from effects.identity import IdentityEffect
from helpers.visual_parameter_def import VisualParameterDef


class EffectBase(torch.nn.Module):
    def __init__(self, vp_ranges):
        super().__init__()
        self.create_checkpoints = False
        self.vpd = VisualParameterDef(vp_ranges)

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
