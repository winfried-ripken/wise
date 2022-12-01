from effects.minimal_pipeline import MinimalPipelineEffect
from effects.xdog import XDoGEffect
from helpers.visual_parameter_def import portrait_preset, add_optional_params, remove_optional_presets, minimal_pipeline_presets, \
    minimal_pipeline_vp_ranges

xdog_params = ["blackness", "contour", "strokeWidth", "details", "saturation", "contrast", "brightness"]
minimal_pipeline_params = [x[0] for x in minimal_pipeline_vp_ranges]



def get_default_settings(name, **kwargs):
    if name == "xdog":
        effect = XDoGEffect(**kwargs)
        presets = portrait_preset
        params = xdog_params
    elif name == "minimal_pipeline":
        # kwargs['enable_adapt_hue_preprocess'] = False
        # kwargs['enable_adapt_hue_postprocess'] = False
        effect = MinimalPipelineEffect()
        presets = minimal_pipeline_presets
        params = minimal_pipeline_params
    else:
        raise ValueError(f"effect {name} not found")
    return effect, remove_optional_presets(presets, **kwargs), add_optional_params(params, **kwargs)
