import argparse
import os
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np

from effects import get_default_settings
from helpers import make_save_path
from helpers.visual_parameter_def import VisualParameterDef
from parameter_optimization.optimization_utils import optimize_single_parameter

CONFIG = {
    "lr_start": 0.1,
    "lr_stop": 0.01,
    "lr_decay": 0.99,
    "lr_decay_start": 50,
    "n_iterations": 200}


def check_single_parameter(module, preset,
                           uniform_name, n_steps=4, n_starts=3,
                           verbose=False, exp_name=None):
    if exp_name is None:
        exp_name = type(module).__name__

    color = plt.cm.rainbow(np.linspace(0, 1, n_starts))
    diffs = []

    save_path = make_save_path(f"{os.path.dirname(__file__)}/../experiments/results/"
                               f"{exp_name}/{uniform_name}")

    uniform_min, uniform_max = VisualParameterDef.get_param_range()
    target_step = 0 if n_starts == 1 else (uniform_max - uniform_min) / (n_starts + 1)
    uniform_target = random.uniform(uniform_min, uniform_max) if n_steps == 1 else uniform_min + target_step

    fig, ax = plt.subplots()
    ax.set_xlim([-0.51, 0.51])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel(f"{uniform_name} parameter")
    ax.set_ylabel(f"difference abs(pred - target)")

    for j in range(n_starts):
        uniform_start = uniform_min + ((uniform_max - uniform_min) / 2.0) if n_steps == 1 else uniform_min
        step = 0 if n_steps == 1 else (uniform_max - uniform_min) / (n_steps - 1)

        xs = []
        ys = []

        for i in range(n_steps):
            diff = optimize_single_parameter(save_path, preset, module, uniform_name, uniform_start, uniform_target,
                                             verbose=verbose, config=CONFIG)

            xs.append(uniform_start)
            ys.append(diff.item())

            uniform_start += step

        ax.plot(xs, ys, color=color[j])
        ax.scatter(xs, ys, color=color[j])
        # ax.axvline(uniform_target, color=color[j])

        diffs += ys
        uniform_target += target_step

    # note that the difference is in normalized space
    if mean(diffs) < 0.1:
        fig.patch.set_facecolor('xkcd:mint green')
    else:
        fig.patch.set_facecolor('xkcd:salmon')

    # SAVING here
    np.save(f"{save_path}/result.npy", np.array(diffs))
    fig.savefig(f"{save_path}/result.png")
    fig.clear()
    plt.close(fig)


def check_all_params(effect, preset, start_idx, stop_idx):
    for n, mi, ma in effect.vpd.vp_ranges[start_idx:stop_idx]:
        check_single_parameter(effect, preset, n, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--effect', help='which effect to use', default="xdog")
    parser.add_argument('--start', help='parameter start index', type=int, default=0)
    parser.add_argument('--stop', help='parameter stop index', type=int, default=None)
    args = parser.parse_args()

    effect, preset, _ = get_default_settings(args.effect)
    start_idx = args.start
    stop_idx = args.stop

    print("effect", args.effect)
    print("start", start_idx)
    print("stop", stop_idx)

    # torch.autograd.set_detect_anomaly(True)  #  SLOW
    check_all_params(effect, preset, start_idx, stop_idx)
