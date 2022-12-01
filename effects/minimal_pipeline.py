import torch

from effects.bump_mapping import BumpMappingEffect
from effects.edge_blend import EdgeBlendEffect
from effects.flow_aligned_smoothing import FlowAlignedSmoothingEffect
from effects.gauss2d_xy_separated import Gauss2DEffect
from effects.lab_to_rgb import LabToRGBEffect
from effects.noise import NoiseEffect
from effects.oilpaint_compose import OilpaintComposeEffect
from effects.rgb_to_lab import RGBToLabEffect
from effects.structure_tensor import StructureTensorEffect
from effects.tangent_flow_map import TangentFlowEffect
from effects.xdog_pass0 import XDoGPass0Effect
from effects.xdog_pass1 import XDoGPass1Effect
from helpers import color_conversion
from helpers.effect_base import EffectBase
from helpers.index_helper import IndexHelper
from helpers.visual_parameter_def import minimal_pipeline_vp_ranges


class MinimalPipelineEffect(EffectBase):
    def __init__(self, **kwargs):
        super().__init__(minimal_pipeline_vp_ranges)
        # locals
        dim_kernsize = 5

        # effects
        self.rgb_to_lab = RGBToLabEffect()
        self.lab_to_rgb = LabToRGBEffect()

        self.structureTensorPass = StructureTensorEffect()
        self.gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=dim_kernsize)
        self.gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=dim_kernsize)
        self.tangent_flow = TangentFlowEffect()

        self.xDoGPass0 = XDoGPass0Effect(dim_kernsize=dim_kernsize)
        self.xDoGPass1 = XDoGPass1Effect()

        self.edge_blend = EdgeBlendEffect()

        self.noise = NoiseEffect()
        self.noise_smoothing = FlowAlignedSmoothingEffect(True)

        self.bump = BumpMappingEffect()

        self.compose = OilpaintComposeEffect()

    def forward_effect(self, x, visual_parameters):
        i = IndexHelper(x)

        contrast = self.vpd.select_parameter(visual_parameters, "contrast")
        colorfulness = self.vpd.select_parameter(visual_parameters, "colorfulness")
        luminosity_offset = self.vpd.select_parameter(visual_parameters, "luminosityOffset")
        hueShift = self.vpd.select_parameter(visual_parameters, "hueShift")
        contour = self.vpd.select_parameter(visual_parameters, "contour")
        contourOpacity = self.vpd.select_parameter(visual_parameters, "contourOpacity")
        bumpScale = self.vpd.select_parameter(visual_parameters, "bumpScale")
        bump_phong_specular = self.vpd.select_parameter(visual_parameters, "bumpSpecular")
        bump_opacity = self.vpd.select_parameter(visual_parameters, "bumpOpacity")

        x = self.luminosity_adjustment(x, luminosity_offset, i)
        colorize = self.color_adjustment_part(x, i, contrast, colorfulness, hueShift)

        gauss_sigma_1 = torch.tensor(0.5251, device=x.device)

        xdog_sigma_narrow = torch.tensor(1, device=x.device)
        xdog_sigma_edge = torch.tensor(2, device=x.device)
        xdog_epsilon = torch.tensor(-0.2, device=x.device)
        xdog_phi = torch.tensor(1.0, device=x.device)

        edge_blend = self.contours_part(self.run(self.rgb_to_lab, colorize), colorize, i, gauss_sigma_1,
                                        xdog_sigma_narrow, xdog_sigma_edge, xdog_epsilon,
                                        xdog_phi, contour, contourOpacity)

        bump = self.paint_texture_part(edge_blend, bumpScale, bump_phong_specular)
        edge_blend = i.cat(i.idx(edge_blend, 'xyz'), bump_opacity)
        output = self.composition_part(bump, edge_blend)
        return output

    def paint_texture_part(self, x, bump_scale, bump_phong_specular):
        noise_smoothing_step_size = torch.tensor(1.0, device=x.device)
        noise_smoothing_step_size_scaling_factor = torch.tensor(0.0, device=x.device)
        colorSmoothing = torch.tensor(10, device=x.device)  # the "Size/details" of noise (smaller looks better)
        tf_1_gauss_sigma = torch.tensor(16.0, device=x.device)  # The smoothness of noise 'strokeLength'

        bump_sample_distance = torch.tensor(0.2626, device=x.device)
        bump_phong_shininess = torch.tensor(14.0, device=x.device)
        brushScale = torch.tensor(1.0, device=x.device)

        noise_smoothing_sigma = colorSmoothing / 3.8086
        bump_sigma_color = colorSmoothing / 3.8086
        noise_scale = brushScale / 3.8086

        precisionFactor = torch.tensor(1.0 / 3.8086, device=x.device)
        structure_tensor_sigma = torch.tensor(1.0 / 3.8086, device=x.device)
        tf_1, _ = self.tf_map(self.run(self.rgb_to_lab, x), structure_tensor_sigma,
                              tf_1_gauss_sigma, precisionFactor)
        noise = self.noise(x, noise_scale)
        noise_smoothing = self.noise_smoothing(noise, tf_1,
                                               noise_smoothing_sigma,
                                               noise_smoothing_step_size,
                                               noise_smoothing_step_size_scaling_factor)

        bump = self.run(self.bump, noise_smoothing, bump_sigma_color, bump_scale, bump_phong_shininess,
                        bump_phong_specular, bump_sample_distance)
        return bump

    def composition_part(self, bump, color_smoothing_upsampled):
        output = self.run(self.compose, color_smoothing_upsampled, bump)
        return output

    def luminosity_adjustment(self, image: torch.Tensor, luminosity_offset: torch.Tensor,
                              i: IndexHelper) -> torch.Tensor:
        lab_image = self.run(self.rgb_to_lab, image)
        i.set_idx(lab_image, 'x', torch.clamp(i.idx(lab_image, 'x') + luminosity_offset, min=0.0, max=1.0))
        return self.run(self.lab_to_rgb, lab_image)

    def contours_part(self, bilateral, colorize, i, gauss_sigma_1,
                      xdog_sigma_narrow, xdog_sigma_edge, xdog_epsilon,
                      xdog_phi, contour, contourOpacity):
        structure_tensor_sigma = torch.tensor(1.0 / 3.8086, device=bilateral.device)
        xdog_wide_kernel_weight = contour / 100.0

        tf, _ = self.tf_map(bilateral, structure_tensor_sigma, gauss_sigma_1, precisionFactor=1.0)
        xdog_contours_intermediate = self.run(self.xDoGPass0, bilateral, tf, xdog_wide_kernel_weight,
                                              xdog_sigma_narrow)
        xdog_contours = self.run(self.xDoGPass1, xdog_contours_intermediate, tf, xdog_epsilon,
                                 xdog_sigma_edge, xdog_phi)
        xdog_contours = i.idx(xdog_contours, "xxx")
        edge_blend = self.run(self.edge_blend, colorize, xdog_contours, contourOpacity)
        return edge_blend

    def color_adjustment_part(self, x, i, contrast, saturation, hueShift):
        x = (x - 0.5) * i.view(contrast) + 0.5
        x = torch.clamp(x, 0.0, 1.0)

        hsv = color_conversion.rgb_to_hsv(x)

        i.set_idx(hsv, "g", torch.clamp(i.idx(hsv, "g") * i.view(saturation), 0.0, 1.0))
        i.set_idx(hsv, "r", torch.frac(i.idx(hsv, "r") + hueShift))

        rgb = color_conversion.hsv_to_rgb(hsv)
        color_adjustment = torch.clamp(rgb, 0.0, 1.0)

        return color_adjustment

    def tf_map(self, xlab, sigma_sst, sigma_gauss, precisionFactor):
        sst = self.run(self.structureTensorPass, xlab, sigma_sst)
        gauss_1 = self.run(self.gauss2dx, sst, sigma_gauss, precisionFactor)
        gauss_2 = self.run(self.gauss2dy, gauss_1, sigma_gauss, precisionFactor)
        return self.run(self.tangent_flow, gauss_2), sst
