from copy import deepcopy
import torch
from helpers.scale_visual_parameters import ScaleVisualParameters

optional_param_map = {
    'enable_depth_enhance': [("depthPower", 0.0, 4.0),
                             ("depthColorTransfer", 0.0, 1.0)],
    'enable_adapt_hue_preprocess': [("adaptHuePreprocess", 0.0, 1.0)],
    'enable_adapt_hue_postprocess': [("adaptHuePostprocess", 0.0, 1.0)]
}

xdog_vp_ranges = [("brightness", -0.15, 0.5),
                  ("contrast", 0.2, 3.0),
                  ("saturation", 0.0, 3.0),
                  ("details", 0.0, 3.0),
                  ("strokeWidth", 0.3, 1.2),
                  ("contour", 0.0, 3.0),
                  ("blackness", 0.01, 0.5)]


portrait_preset = [("brightness", 0.1),
                   ("contrast", 1.15),
                   ("saturation", 1.02),
                   ("strokeWidth", 0.54),
                   ("contour", 1.0),
                   ("details", 2.27),
                   ("blackness", 0.15)]

sketch_preset = [("brightness", 0.1),
                 ("contrast", 1.0),
                 ("saturation", 0.0),
                 ("strokeWidth", 0.2),
                 ("contour", 3.0),
                 ("details", 1.3),
                 ("blackness", 0.5)]

minimal_pipeline_vp_ranges = [("contour", 0.0, 100.0),
                              ("contourOpacity", 0.0, 1.0),
                              ("bumpScale", 0.0, 17.0),
                              ("bumpSpecular", 0.5, 20.0),
                              ("bumpOpacity", 0.0, 1.0),
                              ("colorfulness", -1.4, 2.0),
                              ("luminosityOffset", -1.0, 1.0),
                              ("contrast", 0.5, 4.0),
                              ("hueShift", -2.0, 2.0)]

minimal_pipeline_presets = [("contour", 50),
                            ("contourOpacity", 0.0),
                            ("bumpScale", 3.0),
                            ("bumpSpecular", 1.0),
                            ("bumpOpacity", 0.0),
                            ("colorfulness", 1.5),
                            ("luminosityOffset", 0.0),
                            ("contrast", 1.0),
                            ("hueShift", 0.0)]

minimal_pipeline_bump_mapping_preset = [("contour", 50),
                                        ("contourOpacity", 0.2),
                                        ("bumpScale", 17.0),
                                        ("bumpSpecular", 15.0),
                                        ("bumpOpacity", 0.6),
                                        ("colorfulness", 1.5),
                                        ("luminosityOffset", -0.0),
                                        ("contrast", 1.5),
                                        ("hueShift", 0.0)]

minimal_pipeline_xdog_preset = [("contour", 100),
                                ("contourOpacity", 0.8),
                                ("bumpScale", 3.0),
                                ("bumpSpecular", 1.0),
                                ("bumpOpacity", 0.2),
                                ("colorfulness", 1.5),
                                ("luminosityOffset", -0.1),
                                ("contrast", 1.5),
                                ("hueShift", 0.0)]




def remove_optional_presets(presets, **kwargs):
    presets = deepcopy(presets)
    for optional_param_arg_name, optional_vp_ranges in optional_param_map.items():
        if not kwargs.get(optional_param_arg_name, False):
            preset_names_to_remove = [optional_vp_range[0] for optional_vp_range in optional_vp_ranges]
            presets = list(filter(lambda preset: preset[0] not in preset_names_to_remove, presets))
    return presets


def add_optional_params(params, **kwargs):
    params = deepcopy(params)
    for optional_param_arg_name, optional_vp_ranges in optional_param_map.items():
        if kwargs.get(optional_param_arg_name, False):
            optional_params = [optional_vp_range[0] for optional_vp_range in optional_vp_ranges]
            params.extend(optional_params)
    return params


class VisualParameterDef:
    def __init__(self, vp_ranges):
        self.name2idx = {}
        self.vp_ranges = vp_ranges
        self.scale_parameters = ScaleVisualParameters(vp_ranges)
        # Note that this is the only place where parameters should be really scaled

        i = 0
        for n, _, _ in vp_ranges:
            self.name2idx[n] = i
            i += 1

    def select_parameter(self, tensor, name):
        return tensor[:, self.name2idx[name]:self.name2idx[name] + 1]

    def select_parameters(self, tensor, parameter_names):
        result = tensor.new_empty((tensor.size(0), len(parameter_names), *tensor.size()[2:]))

        for i, pn in enumerate(parameter_names):
            result[:, i] = tensor[:, self.name2idx[pn]]

        return result

    def preset_tensor(self, preset, reference_tensor, add_local_dims=False):
        if add_local_dims:
            dims = (reference_tensor.size(0), len(self.name2idx), reference_tensor.size(2), reference_tensor.size(3))
        else:
            dims = (reference_tensor.size(0), len(self.name2idx))

        result = reference_tensor.new_empty(dims)
        for n, v in preset:
            result[:, self.name2idx[n]] = v

        return self.scale_parameters(result, True)  # scale back

    def update_visual_parameters(self, vp_tensor, parameter_names, update_tensor, support_cascading=False):
        if support_cascading:
            print("CASCADING should not be used anymore")
            raise RuntimeError("cascading")

        vp_tensor = vp_tensor.clone()
        for i, pn in enumerate(parameter_names):
            vp_tensor[:, self.name2idx[pn]] = update_tensor[:, i]

        return vp_tensor

    @staticmethod
    def get_param_range():
        return -0.5, 0.5

    @staticmethod
    def clamp_range(vp):
        return torch.clamp(vp, -0.5, 0.5)

    @staticmethod
    def rand_like(vp):
        return torch.rand_like(vp) - 0.5

    @staticmethod
    def rand(shape):
        return torch.rand(shape) - 0.5
