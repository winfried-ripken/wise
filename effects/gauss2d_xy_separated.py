import torch

from helpers.index_helper import IndexHelper


class Gauss2DEffect(torch.nn.Module):
    def __init__(self, dxdy, dim_kernsize):
        super().__init__()
        self.dxdy = dxdy
        self.dim_kernsize = dim_kernsize

    def forward(self, x, sigma, precisionFactor=1.0):
        i = IndexHelper(x)

        sigma = i.view(sigma, dims=5) * i.input_size_factor() * precisionFactor
        twoSigmaSquare = 2.0 * sigma * sigma
        halfStepWidth = torch.ceil(2.0 * sigma)
        direction = torch.tensor(self.dxdy, device=x.device).reshape(1, 2, 1, 1) / i.tex_size()

        itr = torch.arange(-self.dim_kernsize, self.dim_kernsize + 1, device=x.device).view(1, -1, 1, 1, 1).float()
        gk = torch.exp(-itr * itr / twoSigmaSquare)
        gk = torch.where((itr >= -halfStepWidth).logical_and(itr <= halfStepWidth), gk, torch.zeros_like(gk))

        offsets = itr * direction.unsqueeze(1)
        s = i.sample_complex(x, offsets, add_p_start=True)

        sum = (s * gk).sum(dim=1)
        norm = gk.sum(dim=1)

        return i.safe_div(sum, norm)
