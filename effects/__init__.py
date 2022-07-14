#from effects.toon import ToonEffect
#from effects.watercolor import WatercolorEffect
from effects.xdog import XDoGEffect
from helpers.visual_parameter_def import portrait_preset, coffee_preset, watercolor_vp_ranges

xdog_params = ["blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness"]
toon_params = xdog_params + ["finalSmoothing", "colorBlur", "colorQuantization"]  # this is the color quant. version
watercolor_params = [x[0] for x in watercolor_vp_ranges]


def get_default_settings(name):
    if name == "xdog":
        return XDoGEffect(), portrait_preset, xdog_params
    #elif name == "watercolor":
    #    return WatercolorEffect(), coffee_preset, watercolor_params
    #elif name == "toon":
    #    return ToonEffect(), portrait_preset, toon_params
    else:
        raise ValueError(f"effect {name} not found")
