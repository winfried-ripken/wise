import torch
from helpers.scale_visual_parameters import ScaleVisualParameters

toon_vp_ranges = [("brightness", -0.15, 0.5),
                  ("contrast", 0.2, 3.0),
                  ("saturation", 0.0, 3.0),
                  ("colorQuantization", 5.0, 30.0),
                  ("colorBlur", 0.0125, 0.14),
                  ("details", 0.0, 3.0),
                  ("strokeWidth", 0.3, 1.2),
                  ("contour", 0.0, 3.0),
                  ("blackness", 0.01, 0.5),
                  ("spotColor_r", 0.0, 255.0),
                  ("spotColor_g", 0.0, 255.0),
                  ("spotColor_b", 0.0, 255.0),
                  ("spotColorAmount", 0.0, 1.0),
                  ("finalSmoothing", 1.0, 10.0)]

portrait_preset = [("brightness", 0.1),
                   ("contrast", 1.15),
                   ("saturation", 1.02),
                   ("colorQuantization", 18.93),
                   ("colorBlur", 0.14),
                   ("strokeWidth", 0.54),
                   ("contour", 1.0),
                   ("details", 2.27),
                   ("blackness", 0.15),
                   ("finalSmoothing", 1.0),
                   ("spotColor_r", 153.0),
                   ("spotColor_g", 197.0),
                   ("spotColor_b", 208.0),
                   ("spotColorAmount", 0.0)]

sketch_preset = [("brightness", 0.1),
                 ("contrast", 1.0),
                 ("saturation", 0.0),
                 ("colorQuantization", 5.0),
                 ("colorBlur", 0.0125),
                 ("strokeWidth", 0.2),
                 ("contour", 3.0),
                 ("details", 1.3),
                 ("blackness", 0.5),
                 ("finalSmoothing", 2.0),
                 ("spotColor_r", 153.0),
                 ("spotColor_g", 197.0),
                 ("spotColor_b", 208.0),
                 ("spotColorAmount", 0.0)]

# note that some of the presets are originally not within those ranges
# but it is neccessary for good grads
watercolor_vp_ranges = [("depthPower", 0.0, 4.0),
                        ("depthColorTransfer", 0.0, 1.0),
                        ("colorAmount", 0.5, 0.95),
                        ("paintSplatter", 0.0, 3.0),
                        ("colorfulness", 0.0, 2.0),
                        ("luminosity", 0.3, 4.0),
                        ("contrast", 0.5, 1.8),
                        ("highlights", 0.01, 1.99),
                        ("shadows", -0.99, 0.99),
                        ("blackpoint", -0.5, 0.5),
                        ("edgeDarkening", 25.0, 100.0),
                        ("details", 0.1, 10.0),
                        ("wobbling", 0.0, 10.0),
                        ("wetnessScale", 5.0, 80.0),
                        ("contourWidth", 0.5, 5.0),
                        ("contour", 10.0, 100.0),
                        ("gaps", 10.0, 200.0),
                        ("gapScale", 0.85, 2.3),
                        ("dispersionMedium", 0.0, 10.0),
                        ("dispersionFine", 0.0, 10.0)]

watercolor_IOS_vp_ranges = [("depthPower", 0.0, 4.0),
                            ("depthColorTransfer", 0.0, 1.0),
                            ("colorAmount", 0.0, 1.0),
                            ("paintSplatter", 0.0, 5.0),
                            ("colorfulness", -1.4, 2.0),
                            ("luminosity", 0.3, 4.0),
                            ("contrast", 0.5, 1.8),
                            ("highlights", 0.01, 1.99),
                            ("shadows", -0.99, 0.99),
                            ("blackpoint", -0.5, 0.5),
                            ("edgeDarkening", 0.0, 100.0),
                            ("details", 0.1, 14.0),
                            ("wobbling", 0.0, 10.0),
                            ("wetnessScale", 0.0, 80.0),
                            ("contourWidth", 0.4, 5.0),
                            ("contour", 0.0, 100.0),
                            ("gaps", 0.0, 100.0),
                            ("gapScale", 0.0, 5.0),
                            ("dispersionMedium", 0.0, 10.0),
                            ("dispersionFine", 0.0, 10.0)]
# MISSING PARAMS:
# wobbling scale
# paint splatter scale
# erase color
# fill color
# fill color opacity
# wetness
# contour opacity
# sketchiness
# dispersion medium scale
# dispersion fine scale
# edge bleed
# pre abstraction max dimension = 1024 (?)
# GAPS has larger range! (200) and is used - should we adapt this?
# color lut doesnt exist!

# depthPower: MIN 0.0 MAX 4.0 BASE 2.0 (= 0.0)
# depthColorTransfer: MIN 0.0 MAX 1.0 BASE 0.5 (= 0.0)
# colorAmount: MIN 0.0 MAX 1.0 BASE 0.5 (= 0.0)
# paintSplatter: MIN 0.0 MAX 5.0 BASE 2.5 (= 0.0)
# colorfulness: MIN -1.4 MAX 2.0 BASE 0.30000000000000004 (= 0.0)
# luminosity: MIN 0.3 MAX 4.0 BASE 2.15 (= 0.0)
# contrast: MIN 0.5 MAX 1.8 BASE 1.15 (= 0.0)
# highlights: MIN 0.01 MAX 1.99 BASE 1.0 (= 0.0)
# shadows: MIN -0.99 MAX 0.99 BASE 0.0 (= 0.0)
# blackpoint: MIN -0.5 MAX 0.5 BASE 0.0 (= 0.0)
# edgeDarkening: MIN 0.0 MAX 100.0 BASE 50.0 (= 0.0)
# details: MIN 0.1 MAX 14.0 BASE 7.05 (= 0.0)
# wobbling: MIN 0.0 MAX 10.0 BASE 5.0 (= 0.0)
# wetnessScale: MIN 0.0 MAX 80.0 BASE 40.0 (= 0.0)
# contourWidth: MIN 0.4 MAX 5.0 BASE 2.7 (= 0.0)
# contour: MIN 0.0 MAX 100.0 BASE 50.0 (= 0.0)
# gaps: MIN 0.0 MAX 100.0 BASE 50.0 (= 0.0)
# gapScale: MIN 0.0 MAX 5.0 BASE 2.5 (= 0.0)
# dispersionMedium: MIN 0.0 MAX 10.0 BASE 5.0 (= 0.0)
# dispersionFine: MIN 0.0 MAX 10.0 BASE 5.0 (= 0.0)

coffee_preset = [("depthPower", 3.0),
                 ("depthColorTransfer", 0.0),
                 ("details", 2.0),
                 ("wobbling", 2.0),
                 ("paintSplatter", 3.0),
                 ("gaps", 60.0),
                 ("gapScale", 1.2),
                 ("colorAmount", 1.0),
                 ("colorfulness", 0.0),
                 ("luminosity", 3.0),
                 ("contrast", 1.0),
                 ("shadows", 0.0),
                 ("highlights", 0.7),
                 ("blackpoint", -0.1),
                 ("edgeDarkening", 99.0),
                 ("wetnessScale", 60.0),
                 ("contour", 60.0),
                 ("contourWidth", 1.8),
                 ("dispersionMedium", 1.8),
                 ("dispersionFine", 1.8)]

clean_preset = [("depthPower", 2.0),
                ("depthColorTransfer", 0.0),
                ("details", 6.0),
                ("wobbling", 2.0),
                ("paintSplatter", 2.0),
                ("gaps", 0.0),
                ("gapScale", 1.2),
                ("colorAmount", 1.0),
                ("colorfulness", 0.4),
                ("luminosity", 2.0),
                ("contrast", 1.0),
                ("shadows", 0.0),
                ("highlights", 1.0),
                ("blackpoint", -0.1),
                ("edgeDarkening", 0.0),
                ("wetnessScale", 0.0),
                ("contour", 0.0),
                ("contourWidth", 0.7),
                ("dispersionMedium", 1.0),
                ("dispersionFine", 0.6)]

historic_preset = [("depthPower", 2.0),
                   ("depthColorTransfer", 0.0),
                   ("details", 4.0),
                   ("wobbling", 1.0),
                   ("paintSplatter", 1.5),
                   ("gaps", 20.0),
                   ("gapScale", 2.3),
                   ("colorAmount", 1.0),
                   ("colorfulness", 0.2),
                   ("luminosity", 2.0),
                   ("contrast", 1.0),
                   ("shadows", 0.0),
                   ("highlights", 1.8),
                   ("blackpoint", -0.05),
                   ("edgeDarkening", 99.0),
                   ("wetnessScale", 0.0),
                   ("contour", 19.0),
                   ("contourWidth", 2.0),
                   ("dispersionMedium", 1.0),
                   ("dispersionFine", 0.6)]

luminous_2_preset = [("depthPower", 2.0),
                     ("depthColorTransfer", 0.0),
                     ("details", 7.75),
                     ("wobbling", 2.0),
                     ("paintSplatter", 2.0),
                     ("gaps", 50.0),
                     ("gapScale", 1.0),
                     ("colorAmount", 0.98),
                     ("colorfulness", 2.0),
                     ("luminosity", 4.0),
                     ("contrast", 1.0),
                     ("shadows", 0.0),
                     ("highlights", 1.0),
                     ("blackpoint", -0.05),
                     ("edgeDarkening", 99.0),
                     ("wetnessScale", 30.0),
                     ("contour", 30.0),
                     ("contourWidth", 0.9),
                     ("dispersionMedium", 1.0),
                     ("dispersionFine", 0.6)]

natural_preset = [("depthPower", 2.0),
                  ("depthColorTransfer", 0.0),
                  ("details", 7.75),
                  ("wobbling", 2.0),
                  ("paintSplatter", 2.0),
                  ("gaps", 30.0),
                  ("gapScale", 1.0),
                  ("colorAmount", 1.0),
                  ("colorfulness", 0.5),
                  ("luminosity", 3.0),
                  ("contrast", 1.0),
                  ("shadows", 0.0),
                  ("highlights", 1.0),
                  ("blackpoint", -0.1),
                  ("edgeDarkening", 99.0),
                  ("wetnessScale", 10.0),
                  ("contour", 32.3),
                  ("contourWidth", 1.0),
                  ("dispersionMedium", 1.0),
                  ("dispersionFine", 0.6)]

rainy_preset = [
    ("depthPower", 2.0),
    ("depthColorTransfer", 0.0),
    ("details", 0.0),
    ("wobbling", 2.0),
    ("paintSplatter", 2.0),
    ("gaps", 50.0),
    ("gapScale", 1.0),
    ("colorAmount", 1.0),
    ("colorfulness", 0.0),
    ("luminosity", 3.0),
    ("contrast", 1.0),
    ("shadows", 0.0),
    ("highlights", 1.0),
    ("blackpoint", -0.1),
    ("edgeDarkening", 99.0),
    ("wetnessScale", 20.0),
    ("contour", 99.0),
    ("contourWidth", 1.6),
    ("dispersionMedium", 2.5),
    ("dispersionFine", 0.6)]

soaked_preset = [("depthPower", 1.33),
                 ("depthColorTransfer", 0.0),
                 ("details", 0.0),
                 ("wobbling", 2.0),
                 ("paintSplatter", 1.0),
                 ("gaps", 35.0),
                 ("gapScale", 1.4),
                 ("colorAmount", 1.0),
                 ("colorfulness", 0.8),
                 ("luminosity", 5.0),
                 ("contrast", 1.0),
                 ("shadows", 0.0),
                 ("highlights", 1.0),
                 ("blackpoint", 0.0),
                 ("edgeDarkening", 99.0),
                 ("wetnessScale", 30.0),
                 ("contour", 32.3),
                 ("contourWidth", 2.0),
                 ("dispersionMedium", 1.0),
                 ("dispersionFine", 0.6)]

technical_preset = [("depthPower", 2.0),
                    ("depthColorTransfer", 0.0),
                    ("details", 2.0),
                    ("wobbling", 2.0),
                    ("paintSplatter", 2.0),
                    ("gaps", 50.0),
                    ("gapScale", 1.8),
                    ("colorAmount", 0.97),
                    ("colorfulness", 0.8),
                    ("luminosity", 3.0),
                    ("contrast", 1.0),
                    ("shadows", 0.0),
                    ("highlights", 1.2),
                    ("blackpoint", 0.0),
                    ("edgeDarkening", 99.0),
                    ("wetnessScale", 0.0),
                    ("contour", 80.0),
                    ("contourWidth", 1.4),
                    ("dispersionMedium", 2.0),
                    ("dispersionFine", 1.0)]


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
