import torch

from effects.color_adjustment_toon import ColorAdjustmentToonEffect
from effects.flow_aligned_bilateral import FlowAlignedBilateralEffect
from effects.gauss2d_xy_separated import Gauss2DEffect
from effects.rgb_to_lab import RGBToLabEffect
from effects.structure_tensor import StructureTensorEffect
from effects.tangent_flow_map import TangentFlowEffect
from effects.xdog_pass0 import XDoGPass0Effect
from effects.xdog_pass1 import XDoGPass1Effect
from helpers.effect_base import EffectBase
from helpers.visual_parameter_def import xdog_vp_ranges


class XDoGEffect(EffectBase):
    def __init__(self, repeat_xdog_channel=True, **kwargs):
        super().__init__(xdog_vp_ranges, **kwargs)

        # this parameter controls the latency - accuracy tradeoff
        dim_kernsize = 5
        self.create_checkpoints = False

        self.colorAdjustmentPass = ColorAdjustmentToonEffect()
        self.rgb_to_lab = RGBToLabEffect()
        self.structureTensorPass = StructureTensorEffect()
        self.gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=dim_kernsize)
        self.gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=dim_kernsize)
        self.tangent_flow = TangentFlowEffect()

        self.bilateralPass0 = FlowAlignedBilateralEffect(False, dim_kernsize=dim_kernsize)
        self.bilateralPass1 = FlowAlignedBilateralEffect(True, dim_kernsize=dim_kernsize)

        self.xDoGPass0 = XDoGPass0Effect(dim_kernsize=dim_kernsize)
        self.xDoGPass1 = XDoGPass1Effect()
        self.repeat_xdog_channel = repeat_xdog_channel

    def forward_effect(self, x, visual_parameters):
        dt = self.vpd.select_parameter(visual_parameters, "details")

        cap_brightness = self.vpd.select_parameter(visual_parameters, "brightness")
        cap_contrast = self.vpd.select_parameter(visual_parameters, "contrast")
        cap_saturation = self.vpd.select_parameter(visual_parameters, "saturation")
        cap_gamma = torch.tensor(1.0, device=x.device)
        cap_expose = torch.tensor(1.0, device=x.device)
        cap_hueshift = torch.tensor(0.0, device=x.device)

        smoothing_sigma = torch.tensor(1.5, device=x.device)
        xd1_phi = torch.tensor(10.0, device=x.device)

        sst_sigma = 0.75 * (1.0 - dt / 3.0) + 0.25
        bs0_sigma_d = 2.5 * (3.0 - dt)
        bs0_sigma_r = 2.1428571 * (3.0 - dt)
        bs1_sigma_d = 2.5 * (3.0 - dt)
        bs1_sigma_r = 2.1428571 * (3.0 - dt)
        xd1_sigma_edge = 1.2 * (3.0 - dt) + 1.42
        xd0_sigma_narrow = self.vpd.select_parameter(visual_parameters, "strokeWidth")
        xd0_wide_kernel_weight = self.vpd.select_parameter(visual_parameters, "contour")
        xd1_epsilon = self.vpd.select_parameter(visual_parameters, "blackness")

        x = self.run(self.colorAdjustmentPass, x, cap_brightness, cap_contrast, cap_saturation,
                     cap_gamma, cap_expose, cap_hueshift)
        xlab = self.run(self.rgb_to_lab, x)

        sst = self.run(self.structureTensorPass, xlab, sst_sigma)
        sst = self.run(self.gauss2dx, sst, smoothing_sigma)
        sst = self.run(self.gauss2dy, sst, smoothing_sigma)
        tf = self.run(self.tangent_flow, sst)

        bsi = self.run(self.bilateralPass0, xlab, tf, bs0_sigma_d, bs0_sigma_r)
        bs = self.run(self.bilateralPass1, bsi, tf, bs1_sigma_d, bs1_sigma_r)

        xdog = self.run(self.xDoGPass0, bs, tf, xd0_wide_kernel_weight, xd0_sigma_narrow)
        xDogLIC = self.run(self.xDoGPass1, xdog, tf, xd1_epsilon, xd1_sigma_edge, xd1_phi)

        return 1.0 - xDogLIC
