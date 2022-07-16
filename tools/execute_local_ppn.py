import argparse
from pathlib import Path

import torch.nn.functional as F
import torch

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torchvision.transforms import ToPILImage

from effects import xdog_params
from effects.xdog import XDoGEffect
from helpers import torch_to_np, load_image
from helpers.apply_visual_effect import ApplyVisualEffect


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='which jit model to use', default="trained_models/local/abstract_face_ts_model.pt")
    parser.add_argument('--file', help='which input file to use', default="experiments/source/portrait.png")
    args = parser.parse_args()

    m = torch.jit.load(args.model, map_location="cpu")
    m.eval().cuda()

    effect = ApplyVisualEffect(effect_type=XDoGEffect).cuda()
    xxx = load_image(args.file)#.cpu()
    xxx = F.interpolate(xxx, (512, 512))

    test = m(xxx)

    _, ax = plt.subplots(3, 3, figsize=(9, 10.5))

    ax[0, 0].axis("off")
    ax[1, 0].axis("off")
    ax[0, 0].set_title("default output")
    ax[1, 0].set_title("adapted output")

    test_i = effect(xxx, test)
    default = effect.effect.vpd.preset_tensor(effect.default_preset, xxx, False)
    test_s = effect.effect(xxx, default)

    ax[0, 0].imshow(ToPILImage()(test_s.squeeze(0)))
    ax[1, 0].imshow(ToPILImage()(test_i.squeeze(0)))

    for i in range(2, len(xdog_params) + 2):
        im = ax[i % 3, i // 3].imshow(torch_to_np(test[0, i - 2]))
        ax[i % 3, i // 3].axis("off")
        ax[i % 3, i // 3].set_title(xdog_params[i - 2])
        # ax[i % 3, i // 3].colorbar()

        divider = make_axes_locatable(ax[i % 3, i // 3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    Path("experiments/content_adaptive/").mkdir(exist_ok=True)

    plt.tight_layout()
    plt.savefig(f"experiments/content_adaptive/test.png")
    plt.show()
