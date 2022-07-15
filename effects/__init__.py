from effects.xdog import XDoGEffect
from helpers.visual_parameter_def import portrait_preset

xdog_params = ["blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness"]


def get_default_settings(name):
    if name == "xdog":
        return XDoGEffect(), portrait_preset, xdog_params
    else:
        raise ValueError(f"effect {name} not found")
