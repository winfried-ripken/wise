from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor


def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)), Image.BICUBIC)
    return resized


def load_image(path, long_edge=None, cuda=True):
    img = Image.open(path).convert("RGB")
    if long_edge is not None:
        img = pil_resize_long_edge_to(img, long_edge)

    return np_to_torch(img) if not cuda else np_to_torch(img).cuda()


def save_as_image(tensor, path, clamp=False):
    if isinstance(tensor, Tensor):
        tensor = torch_to_np(tensor, clamp)

    Image.fromarray((tensor * 255.0).astype(np.uint8)).save(path)


def np_to_torch(img, add_batch_dim=True, divide_by_255=True):
    img = np.asarray(img).astype(np.float32)

    if len(img.shape) >= 3:
        img = img.transpose((2, 0, 1))
    if add_batch_dim:
        img = img[np.newaxis, ...]

    img_torch = torch.from_numpy(img)
    if img_torch.max() > 1.0 and divide_by_255:
        img_torch = img_torch / 255.0

    return img_torch


def torch_to_np(tensor, clamp=False, multiply_by_255=False):
    tensor = tensor.detach().squeeze().cpu().numpy()

    if clamp:
        tensor = np.clip(tensor, 0.0, 1.0)

    if multiply_by_255:
        tensor *= 255.0

    if len(tensor.shape) < 3:
        return tensor
    elif len(tensor.shape) == 3:
        return tensor.transpose(1, 2, 0)

    return np.moveaxis(tensor, 1, -1)


def resize_keep_aspect_ratio(x, max_size=480):
    h, w = x.size(2), x.size(3)

    # resize for saving computation
    if w > h:
        x = F.interpolate(x, (max_size * h // w, max_size))
    else:
        x = F.interpolate(x, (max_size, max_size * w // h))

    return x


def torch_multiple_and(*tensors):
    result = tensors[0]
    for t in tensors[1:]:
        result = torch.logical_and(result, t)

    return result


def mask_tensor(tensor, active_region):
    return torch.where(active_region, tensor, torch.zeros_like(tensor))


def mask_tensors(tensors, active_region):
    result = []

    for t in tensors:
        result.append(mask_tensor(t, active_region))

    return result


def make_save_path(path, change_fname__if_exists=True):
    Path(path).parent.mkdir(exist_ok=True, parents=True)

    i = 1
    ppp = path
    while change_fname__if_exists and Path(ppp).exists():
        ppp = str(Path(path).parent / (Path(path).stem + f"_{i}" + Path(path).suffix))
        i += 1

    return ppp
