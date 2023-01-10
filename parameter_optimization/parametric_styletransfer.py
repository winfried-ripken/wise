import argparse
import os

import imageio
import torch
import torch.nn.functional as F
from PIL import Image

from pathlib import Path

from effects import get_default_settings, XDoGEffect
from effects.gauss2d_xy_separated import Gauss2DEffect
from helpers import load_image, save_as_image
from helpers.losses import loss_from_string
from helpers.visual_parameter_def import portrait_preset
from parameter_optimization.optimization_utils import run_optimization_loop, ParameterContainer
from parameter_optimization.strotss_org import execute_style_transfer

CONFIG = {
    "lr_start": 0.01,
    "lr_stop": 0.00005,
    "lr_decay": 0.98,
    "lr_decay_start": 50,
    "n_iterations": 1000,
    "local_params": True,
    "sigma_large": 1.5,
    # "smoothing_steps": [10, 25, 50, 100, 250, 500]}
    "smoothing_steps": [10, 25, 50, 100, 175, 250]}


# "smoothing_steps": []}


def single_optimize(module, preset, loss_name, s, t,
                    write_video=False, base_dir=f"{os.path.dirname(__file__)}/../experiments/result", prepare_effect=None,
                    smoothing=True, cpu=False, iter_callback=lambda step: None):
    loss_f = loss_name if isinstance(loss_name, torch.nn.Module) else loss_from_string(loss_name)
    Path(base_dir).mkdir(exist_ok=True, parents=True)

    output_name = f"{base_dir}/OUT_{Path(s).name[:5]}_{loss_name}"
    batch_im = load_image(s, cuda=not cpu)
    targets = load_image(t, cuda=not cpu)
    batch_im = F.interpolate(batch_im, (targets.size(2), targets.size(3)))

    if isinstance(module, XDoGEffect):
        module.smoothing_sigma = 1.3
        module.xd1_phi = 1.0

    module.enable_checkpoints()
    module = module if cpu else module.cuda()

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
            sigma_large = torch.tensor(CONFIG["sigma_large"])
            if not cpu:
                sigma_large = sigma_large.cuda()

            vp_smoothed = gauss2dx(grad_vp.vp.data, sigma_large)
            grad_vp.vp.data = gauss2dy(vp_smoothed, sigma_large)
        iter_callback(i)
  
    result, _ = run_optimization_loop(module, grad_vp, batch_im, targets, verbose=True, loss_f=loss_f, config=CONFIG,
                                      vid=writer, callback=cbck, device="cpu" if cpu else "cuda:0")

    if writer is not None:
        writer.close()

    save_as_image(result, f"{output_name}.png", clamp=True)

    # save the parameter maps
    xxx = grad_vp()
    torch.save(xxx.detach().clone(), f"{output_name}.pt")
    return xxx, batch_im


def strotss_process(s, t, base_dir=f"{os.path.dirname(__file__)}/../experiments/result",
                    resize_dim=1024, effect=XDoGEffect(),
                    preset=portrait_preset,
                    cpu=False):

    Path(base_dir).mkdir(exist_ok=True, parents=True)
    base_dir = Path(base_dir)
    strotss_out = base_dir / ("str_" + Path(s).name)

    if not Path(strotss_out).exists():
        result = execute_style_transfer(s, t, resize_dim, device="cpu" if cpu else "cuda:0")
        result.save(strotss_out)

    single_optimize(effect, preset, "l1", s, str(strotss_out), write_video=True,
                    base_dir=str(base_dir), cpu=cpu)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--effect', help='which effect to use', default="xdog")
    parser.add_argument('--content', help='content image', default=f"{os.path.dirname(__file__)}/../experiments/source/portrait.png")
    parser.add_argument('--style', help='style image', default=f"{os.path.dirname(__file__)}/../experiments/target/xdog_portrait.jpg")
    parser.add_argument('--cpu', help='run on cpu', dest="cpu", action="store_true")
    parser.set_defaults(cpu=False)

    args = parser.parse_args()
    effect, preset, _ = get_default_settings(args.effect)
    effect.enable_checkpoints()
    strotss_process(args.content, args.style, effect=effect, preset=preset, cpu=args.cpu)
