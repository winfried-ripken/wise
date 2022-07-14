import argparse
import os

import imageio
import torch
import torch.nn.functional as F

from pathlib import Path

from effects import get_default_settings, XDoGEffect
from effects.gauss2d_xy_separated import Gauss2DEffect
from helpers import load_image_cuda, save_as_image
from helpers.losses import loss_from_string
from helpers.visual_parameter_def import coffee_preset
from parameter_optimization.optimization_utils import run_optimization_loop, ParameterContainer
from parameter_optimization.strotss_org import execute_style_transfer

CONFIG = {
    "lr_start": 0.1,
    "lr_stop": 0.00005,
    "lr_decay": 0.98,
    "lr_decay_start": 50,
    "n_iterations": 500,
    "local_params": True, # use parameter masks
    "sigma_large": 1.3, # kernel for parameter smoothing
    "tvl_factor": 100.0, # strength for total variation loss on parameters
    "smoothing_steps": [10, 25, 50, 100, 250, 500] # smooth at these iterations
    }


def single_optimize(module, preset, loss_name, s, t,
                    write_video=False, base_dir=f"{os.path.dirname(__file__)}/../experiments/result", prepare_effect=None,
                    smoothing=True):
    loss_f = loss_from_string(loss_name)
    Path(base_dir).mkdir(exist_ok=True, parents=True)

    output_name = f"{base_dir}/OUT_{Path(s).name[:5]}_{loss_name}"
    batch_im = load_image_cuda(s)
    targets = load_image_cuda(t)
    batch_im = F.interpolate(batch_im, (targets.size(2), targets.size(3)))

    #if isinstance(module, WatercolorEffect):
    #    print("watercolor effect")
    #    module.preprocess = AdaptHueEffect(batch_im, smooth=False)
    if isinstance(module, XDoGEffect):
        module.smoothing_sigma = 1.3
        module.xd1_phi = 1.0

    module.enable_checkpoints()
    module = module.cuda()

    if prepare_effect is not None:
        module = prepare_effect(module)

    vp = module.vpd.preset_tensor(preset, batch_im, CONFIG["local_params"])
    grad_vp = ParameterContainer(vp, smooth=False)

    writer = imageio.get_writer(f"{output_name}_video.mp4", fps=30) if write_video else None
    gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=5)
    gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=5)

    def cbck(loss, out, lr, i):
        if i in CONFIG["smoothing_steps"] and smoothing and CONFIG["local_params"]:
            # smooth the parameters every few iterations
            # this should decrease artifacts
            vp_smoothed = gauss2dx(grad_vp.vp.data, torch.tensor(CONFIG["sigma_large"]).cuda())
            grad_vp.vp.data = gauss2dy(vp_smoothed, torch.tensor(CONFIG["sigma_large"]).cuda())
  
    result, _ = run_optimization_loop(module, grad_vp, batch_im, targets, verbose=True, loss_f=loss_f, config=CONFIG,
                                      vid=writer, callback=cbck)

    if writer is not None:
        writer.close()

    save_as_image(result, f"{output_name}.png", clamp=True)

    #if isinstance(module, WatercolorEffect):
    #    save_as_image(module.preprocess(batch_im), f"{output_name}_input.png", clamp=True)
    #else:
    #    save_as_image(batch_im, f"{output_name}_input.png", clamp=True)

    # save the parameter maps
    xxx = grad_vp()
    torch.save(xxx.detach().clone(), f"{output_name}.pt")


def strotss_process(s, t, base_dir=f"{os.path.dirname(__file__)}/../experiments/result",
                    resize_dim=1024, effect=XDoGEffect(),
                    preset=coffee_preset):

    Path(base_dir).mkdir(exist_ok=True, parents=True)
    base_dir = Path(base_dir)
    strotss_out = base_dir / ("str_" + Path(s).name)

    if not Path(strotss_out).exists():
        result = execute_style_transfer(s, t, resize_dim)
        result.save(strotss_out)

    single_optimize(effect, preset, "l1", s, str(strotss_out), write_video=True,
                    base_dir=str(base_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--effect', help='which effect to use', default="xdog")
    parser.add_argument('--content', help='content image', default=f"{os.path.dirname(__file__)}/../experiments/source/portrait.png")
    parser.add_argument('--style', help='style image', default=f"{os.path.dirname(__file__)}/../experiments/target/xdog_portrait.jpg")

    args = parser.parse_args()
    effect, preset, _ = get_default_settings(args.effect)
    strotss_process(args.content, args.style, effect=effect, preset=preset)
