import torch

from helpers.index_helper import IndexHelper


class FlowAlignedBilateralEffect(torch.nn.Module):
    def __init__(self, u_pass=True, dim_kernsize=5):
        super().__init__()

        # self.add_uniform("sigmaD", torch.tensor(1.823))
        # self.add_uniform("sigmaR", torch.tensor(1.57))

        self.u_pass = u_pass
        self.dim_kernsize = dim_kernsize

    def forward(self, x, tangent, sigmaD, sigmaR, guide=None):
        i = IndexHelper(x)
        x, x_a = i.get_rgb_and_alpha(x)
        bs = tangent.shape[0]
        image_height = tangent.shape[2]
        image_width = tangent.shape[3]

        if self.u_pass:
            direction_full = tangent.clone()
        else:
            direction_full = i.vec2(i.idx(tangent.clone(), "y"), -i.idx(tangent.clone(), "x"))

        direction_abs = torch.abs(direction_full)
        max_extend = torch.max(direction_abs, dim=1).values
        ds = torch.where(max_extend > 0, 1.0 / max_extend, torch.ones_like(max_extend))

        direction = direction_full / i.tex_size()

        sigmaD = i.view(sigmaD, dims=5) * i.input_size_factor()
        sigmaR = i.view(sigmaR, dims=5)
        twoSigmaDSquare = 2.0 * sigmaD * sigmaD
        twoSigmaRSquare = 2.0 * sigmaR * sigmaR
        halfStepWidth = 2.0 * sigmaD

        sum = x.clone()
        norm = i.const(1.0)

        def compute_iterator(start_idx=1):
            iterator = torch.arange(start_idx, self.dim_kernsize + start_idx, 2, device=x.device).view(1, -1, 1, 1, 1)
            iterator_fw = ds.view(bs, 1, 1, image_height, image_width).repeat(1, iterator.shape[1], 1, 1, 1) * iterator
            return iterator_fw.float()

        itr_d1 = compute_iterator(1)
        itr_d2 = compute_iterator(2)

        def get_kernel(itr, twosigmasquare):
            return torch.exp((-itr * itr) / (twosigmasquare + torch.finfo(x.dtype).eps))

        kD1 = get_kernel(itr_d1, twoSigmaDSquare)
        kD2 = get_kernel(itr_d2, twoSigmaDSquare)

        kernelD = kD1 + kD2

        # clamp with half step width to avoid computing outside the for loop
        kernelD = torch.where(torch.abs(itr_d1) <= halfStepWidth, kernelD, torch.zeros_like(kernelD))

        # avoid division by zero
        itr_d = (itr_d1 * kD1 + itr_d2 * kD2) / (kernelD + torch.finfo(x.dtype).eps)

        c0 = i.sample_complex(x, itr_d * direction.unsqueeze(1), add_p_start=True)
        c1 = i.sample_complex(x, -itr_d * direction.unsqueeze(1), add_p_start=True)

        if guide is None:
            g0 = c0
            g1 = c1
            guide = x
        else:
            g0 = i.sample_complex(guide, itr_d * direction.unsqueeze(1), add_p_start=True)
            g1 = i.sample_complex(guide, -itr_d * direction.unsqueeze(1), add_p_start=True)

        cc0 = (g0 - guide.unsqueeze(1)) * i.vec3_const(100.0, 254.0, 254.0).unsqueeze(1)
        cc1 = (g1 - guide.unsqueeze(1)) * i.vec3_const(100.0, 254.0, 254.0).unsqueeze(1)

        e0 = i.get_len(cc0, dim=2)
        e1 = i.get_len(cc1, dim=2)

        kE0 = get_kernel(e0, twoSigmaRSquare)
        kE1 = get_kernel(e1, twoSigmaRSquare)

        norm += (kernelD * kE0).sum(dim=1)
        norm += (kernelD * kE1).sum(dim=1)

        sum += ((kernelD * kE0).repeat(1, 1, x.size(1), 1, 1) * c0).sum(dim=1)
        sum += ((kernelD * kE1).repeat(1, 1, x.size(1), 1, 1) * c1).sum(dim=1)

        res = sum / norm
        if x_a is not None:
            res = i.cat(res, x_a)
        return res
