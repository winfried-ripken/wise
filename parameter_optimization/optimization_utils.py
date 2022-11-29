import os
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import Module, Parameter
from torch.nn.functional import mse_loss
from torch.optim import Adam
from tqdm.auto import tqdm

from effects.gauss2d_xy_separated import Gauss2DEffect
from helpers import np_to_torch, make_save_path, torch_to_np
from helpers.visual_parameter_def import VisualParameterDef

CONFIG = {
    "lr_start": 0.1,
    "lr_stop": 0.001,
    "lr_decay": 0.9,
    "lr_decay_start": 10,
    "n_iterations": 150}


def run_optimization_loop(module, vp_grad, data_in, target, verbose=True, loss_f=mse_loss,
                          config=None, vp_aux_loss=None, vid=None, callback=None, device="cuda:0"):
    if config is None:
        config = CONFIG

    print("Starting experiment")
    print(config)

    module.train()
    module.to(device)
    data_in = data_in.to(device)
    target = target.to(device)


    if isinstance(loss_f, torch.nn.Module):
        loss_f = loss_f.to(device)

    lr = config["lr_start"]
    optimizer = Adam(list(vp_grad.parameters()) + list(module.parameters()), lr=lr)
    avg_grads = {}
    all_p_values = []

    for i in tqdm(range(config["n_iterations"])):
        if i % 5 == 0 and i > config["lr_decay_start"]:
            lr = lr * config["lr_decay"]

            if lr < config["lr_stop"]:
                lr = config["lr_stop"]

            # decay learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        out = module(data_in, vp_grad())
        loss = loss_f(out, target)

        if vp_aux_loss is not None:
            loss += vp_aux_loss(vp_grad())

        if callback is not None:
            callback(loss, out, lr, i)

        optimizer.zero_grad()
        loss.backward()

        if vid is not None:
            vid.append_data((torch_to_np(out) * 255.0).astype(np.uint8))

        max_grad = 0.0
        p_values = {}
        for name, param in vp_grad.named_parameters():
            if param.grad is None:
                # not needed for computation
                continue

            if name not in avg_grads:
                avg_grads[name] = []

            avg_grads[name].append(param.grad.abs().max().item())
            max_grad = max(param.grad.abs().max().item(), max_grad)
            p_values[name] = param.data.mean().item()  # this makes only sense for 1 el. tensors really

        # ensure that the gradients are proper by printing the largest ones
        if verbose and i % 15 == 0:
            print(f"{i}: loss {loss.item()}. max abs. grad: {max_grad}")
            print(f"parameter values: {p_values}")

        optimizer.step()
        all_p_values.append(p_values)

    out = module(data_in, vp_grad())

    if verbose:
        for name, param in vp_grad.named_parameters():
            if param.grad is None:
                # this guy was not updated
                print(f"{name}: not updated")
                continue

            print(f"{name}: {param.mean().item():.3f}. mean grad: {mean(avg_grads[name])}")
            # this makes only sense for 1 el. tensors really

    return torch_to_np(out), all_p_values


def save_image(result, path, change_fname__if_exists=False):
    if len(result.shape) == 4:
        result = result[0]

    # path = make_save_path(path, change_fname__if_exists)
    test = Image.fromarray(np.abs(result * 255).transpose(1, 2, 0).astype(np.uint8))
    test.save(path)
    print("image saved")


def get_img(name_in="1.png"):
    img = Image.open(f"{os.path.dirname(__file__)}/../experiments/test_data/{name_in}").convert("RGB")
    img = img.resize(CONFIG["resize_shape"])
    return np_to_torch(img)


def get_multiple_images():
    # LEGACY CODE
    inputs = []
    pngs = Path(f"{os.path.dirname(__file__)}/../experiments/test_data/").glob("*.png")

    for im_path in pngs:
        im = Image.open(im_path)
        im = im.resize(CONFIG["resize_shape"])
        inputs.append(np_to_torch(im))

    return torch.cat(inputs, dim=0)


def save_comparison(result, target, save_path):
    f, ax = plt.subplots(2, figsize=(3, 6))
    ax[0].set_title("Result")
    ax[1].set_title("Target")
    ax[0].axis("off")
    ax[1].axis("off")

    ax[0].imshow(result)
    ax[1].imshow(target)

    plt.savefig(save_path)
    plt.close(f)


def optimize_multiple_images(target_module, vp_target, vp_grad,
                             verbose=True, save_path=None, config=None):
    inputs = Image.open(f"{os.path.dirname(__file__)}/../experiments/source/portrait.png")
    inputs = np_to_torch(inputs)

    inputs = inputs.cuda().repeat(vp_target.size(0), 1, 1, 1)
    target_module.cuda()
    vp_target = vp_target.cuda()
    vp_grad = vp_grad.cuda()

    with torch.no_grad():
        target_module.eval()
        targets = target_module(inputs, vp_target)

    result, all_p_values = run_optimization_loop(target_module, vp_grad, inputs, targets, verbose, config=config)

    if verbose:
        save_comparison(result, torch_to_np(targets), make_save_path(f"{save_path}/out.png"))

    return all_p_values


def find_parameter(module, name):
    for n, param in module.named_parameters():
        if n == name:
            return param


class ParameterContainer(Module):
    def __init__(self, vp, repeat_batch_dim=1, smooth=False):
        super().__init__()
        self.vp = Parameter(vp)
        self.repeat_batch_dim = repeat_batch_dim

        self.smooth = smooth
        self.gauss2dx = Gauss2DEffect(dxdy=[1.0, 0.0], dim_kernsize=5)
        self.gauss2dy = Gauss2DEffect(dxdy=[0.0, 1.0], dim_kernsize=5)
        self.sigma = 2.5

    def forward(self):
        self.vp.data = VisualParameterDef.clamp_range(self.vp)
        vp_smoothed = self.vp

        # important that we do not return the clamped tensor directly
        # otherwise the grads will be 0 if we go outside the border

        if self.smooth:
            vp_smoothed = self.gauss2dx(vp_smoothed, torch.tensor(self.sigma).cuda())
            vp_smoothed = self.gauss2dy(vp_smoothed, torch.tensor(self.sigma).cuda())

        return torch.cat(self.repeat_batch_dim*[vp_smoothed], dim=0)


class UpdateSingleVPContainer(Module):
    def __init__(self, vpd, vp, name, p_start, verbose=False):
        super().__init__()

        self.verbose = verbose
        self.vpd = vpd
        self.vp = vp
        self.name = name

        print(f"uniform start {p_start}")
        p_start_tensor = torch.ones_like(vpd.select_parameter(vp, name)) * p_start
        setattr(self, self.name, Parameter(p_start_tensor.squeeze(1)))

    def forward(self):
        our_param = self.vpd.clamp_range(getattr(self, self.name))
        getattr(self, self.name).data = our_param

        # important that we do not return the clamped tensor directly
        # otherwise the grads will be 0 if we go outside the border

        result = self.vp.clone()
        result[:, self.vpd.name2idx[self.name]] = getattr(self, self.name)

        if self.verbose:
            print(f"{self.name}: {result[:, self.vpd.name2idx[self.name]]}")

        return result


def optimize_single_parameter(save_path, preset, target_module,
                              uniform_name="num_bins", uniform_start=5.0,
                              uniform_target=10.0,
                              verbose=True, config=None, ymin=-0.51, ymax=0.51):
    n_trials = 1
    target_module.enable_checkpoints()

    vp = target_module.vpd.preset_tensor(preset, torch.zeros((n_trials, 1)), False).cuda()
    p_idx = target_module.vpd.name2idx[uniform_name]
    vp_target = vp.clone()
    vp_target[:, p_idx] = uniform_target

    vp_grad = UpdateSingleVPContainer(target_module.vpd, vp, uniform_name, uniform_start, verbose=verbose)
    print(f"uniform target {uniform_target})")

    all_p_values = optimize_multiple_images(target_module, vp_target, vp_grad, verbose,
                                            f"{save_path}/img/", config=config)
    if verbose:
        # there should be only one parameter which requires_grad
        # save the graph (parameter over time)
        pname = list(all_p_values[0].keys())[0]
        data = []
        for it in all_p_values:
            data.append(it[pname])

        # SAVING here
        # path = make_save_path(f"{pname}.png", change_fname__if_exists=True)

        fig, ax = plt.subplots()
        ax.plot(data)
        ax.axhline(uniform_target)
        ax.set_ylim([ymin, ymax])
        ax.set_ylabel(f"{uniform_name} parameter")
        ax.set_xlabel(f"iteration")

        fig.savefig(make_save_path(f"{save_path}/parameter_plot.png"))
        fig.clear()
        plt.close(fig)

    # return difference between target and start
    return torch.abs(torch.tensor(uniform_target) - getattr(vp_grad, uniform_name).data.squeeze())
