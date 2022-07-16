import torch
from helpers.scale_visual_parameters import ScaleVisualParameters

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
