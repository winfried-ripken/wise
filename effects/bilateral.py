import torch

from helpers import torch_multiple_and
from helpers.index_helper import IndexHelper


class BilateralEffect(torch.nn.Module):
    def __init__(self, dim_kernsize):
        super().__init__()
        self.dim_kernsize = dim_kernsize

    def forward(self, x, sigmaD, sigmaR):
        i = IndexHelper(x)

        sigmaD = i.view(sigmaD, dims=5) * i.input_size_factor()
        sigmaR = i.view(sigmaR, dims=5)
        twoSigmaDSquare = 2.0 * sigmaD * sigmaD
        twoSigmaRSquare = 2.0 * sigmaR * sigmaR
        halfStepWidth = torch.ceil(2.0 * sigmaD)

        sum = x.clone()
        norm = i.const(1.0)

        # if we used a for loop instead we would really save computations as some zeros would fall out
        # but the gradients for step width could probably break?
        # or are the others from the kernel enough?
        # also: local parameters dont work with for loop obviously... (?)

        it_i = torch.arange(1, self.dim_kernsize + 1, device=x.device).float()
        it_j = torch.arange(-self.dim_kernsize, self.dim_kernsize, device=x.device).float()
        grid_i, grid_j = torch.meshgrid(it_i, it_j)
        grid = torch.stack([torch.flatten(grid_i), torch.flatten(grid_j)]).view(1, 2, -1, 1, 1)

        active_region = torch_multiple_and(grid[:, 0] <= halfStepWidth,
                                           grid[:, 1] >= -halfStepWidth,
                                           grid[:, 1] <= halfStepWidth)
        center_unpacked = x - i.vec3_const(0.0, 0.5, 0.5)  # note that we dont have alpha

        d = i.get_len(grid)
        delta = grid / i.tex_size().unsqueeze(2)
        c0 = i.sample_complex(x, torch.transpose(delta, 2, 1), add_p_start=True).transpose(2, 1)
        c1 = i.sample_complex(x, -torch.transpose(delta, 2, 1), add_p_start=True).transpose(2, 1)

        e0 = (center_unpacked.unsqueeze(2) - (c0 - i.vec3_const(0.0, 0.5, 0.5).unsqueeze(2))) * i.vec3_const(100.0, 254.0, 254.0).unsqueeze(2)
        e1 = (center_unpacked.unsqueeze(2) - (c1 - i.vec3_const(0.0, 0.5, 0.5).unsqueeze(2))) * i.vec3_const(100.0, 254.0, 254.0).unsqueeze(2)

        # check dims
        kernelD = torch.exp(-i.dot(d, d) / (twoSigmaDSquare + torch.finfo(x.dtype).eps))
        kernelE0 = torch.exp(-i.dot(e0, e0) / (twoSigmaRSquare + torch.finfo(x.dtype).eps))
        kernelE1 = torch.exp(-i.dot(e1, e1) / (twoSigmaRSquare + torch.finfo(x.dtype).eps))

        kernelD = torch.where(active_region, kernelD, torch.zeros_like(kernelD))

        norm += (kernelD * kernelE0).sum(dim=2)
        norm += (kernelD * kernelE1).sum(dim=2)
        sum += (kernelD * kernelE0 * c0).sum(dim=2)
        sum += (kernelD * kernelE1 * c1).sum(dim=2)

        return sum / norm
