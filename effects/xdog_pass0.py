import torch

from helpers.index_helper import IndexHelper


class XDoGPass0Effect(torch.nn.Module):
    def __init__(self, dim_kernsize):
        super().__init__()

        self.sigma_wide_to_narrow_ratio = 1.6
        self.dim_kernsize = dim_kernsize + 5

    def forward(self, x, tangent, wide_kernel_weight, sigma_narrow):
        i = IndexHelper(x)
        bs = tangent.shape[0]
        image_height = tangent.shape[2]
        image_width = tangent.shape[3]

        x = i.idx(x, "x")

        grad_full = i.idx(torch.zeros_like(tangent), "xy")
        i.set_idx(grad_full, "y", -i.idx(tangent.clone(), "x"))
        i.set_idx(grad_full, "x", i.idx(tangent.clone(), "y"))

        nabs = torch.abs(grad_full)
        max_extend = torch.max(nabs, dim=1).values
        ds = torch.where(max_extend > 0, 1.0 / max_extend, torch.ones_like(max_extend))

        grad = grad_full / i.tex_size()

        sigmaNarrow = i.view(sigma_narrow, dims=5) * i.input_size_factor()
        sigmaWide = sigmaNarrow * self.sigma_wide_to_narrow_ratio
        halfStepWidth = 2.0 * sigmaWide

        def compute_iterator(start_idx=1):
            itr = torch.arange(start_idx, self.dim_kernsize + start_idx, 2, device=x.device).view(1, -1, 1, 1, 1)
            iterator_fw = ds.view(bs, 1, 1, image_height, image_width).repeat(1, itr.shape[1], 1, 1, 1) * itr
            iterator_bw = ds.view(bs, 1, 1, image_height, image_width).repeat(1, itr.shape[1], 1, 1, 1) * (-itr)
            return torch.cat([iterator_fw, iterator_bw], dim=1).float()

        itr_d1 = compute_iterator(1)
        itr_d2 = compute_iterator(2)

        def get_kernels(itr):
            # we have different kernels for the different pixels as the step width differs
            gw = torch.exp(-itr * itr / (2 * sigmaWide * sigmaWide + torch.finfo(x.dtype).eps))
            gn = torch.exp(-itr * itr / (2 * sigmaNarrow * sigmaNarrow + torch.finfo(x.dtype).eps))

            return gw, gn

        gw_d1, gn_d1 = get_kernels(itr_d1)
        gw_d2, gn_d2 = get_kernels(itr_d2)
        gauss_wide, gauss_narrow = gw_d1 + gw_d2, gn_d1 + gn_d2

        # OpenGL: float d = (d1 * length(kernel1) + d2 * length(kernel2)) / length(kernel);
        len_k1 = torch.sqrt(torch.square(gw_d1) + torch.square(gn_d1) + torch.finfo(x.dtype).eps)
        len_k2 = torch.sqrt(torch.square(gw_d2) + torch.square(gn_d2) + torch.finfo(x.dtype).eps)
        len_k = torch.sqrt(torch.square(gauss_wide) + torch.square(gauss_narrow) + torch.finfo(x.dtype).eps)

        itr_d = (itr_d1 * len_k1 + itr_d2 * len_k2) / (len_k + torch.finfo(x.dtype).eps)
        delta = torch.min(halfStepWidth + 2 * ds - itr_d1, 2 * ds) / (2 * ds)

        # Clamping the iterator
        gauss_wide = torch.where(torch.abs(itr_d1) <= halfStepWidth + 2 * ds, gauss_wide,
                                 torch.zeros_like(gauss_wide))
        gauss_narrow = torch.where(torch.abs(itr_d1) <= halfStepWidth + 2 * ds, gauss_narrow,
                                   torch.zeros_like(gauss_narrow))

        # multiply kernels with delta before summing and normalizing
        gauss_wide = gauss_wide * delta
        gauss_narrow = gauss_narrow * delta

        tex_samples = i.sample_complex(x, itr_d * grad.unsqueeze(1), add_p_start=True)

        # add x for pixels themselves
        result_wide = (tex_samples * gauss_wide).sum(dim=1)
        result_narrow = (tex_samples * gauss_narrow).sum(dim=1)

        wide_norm = gauss_wide.sum(dim=1)
        narrow_norm = gauss_narrow.sum(dim=1)

        wide_norm += 1
        narrow_norm += 1
        result_wide += x
        result_narrow += x

        # divide by norm
        result_wide /= wide_norm
        result_narrow /= narrow_norm

        p = 100 * i.view(wide_kernel_weight)
        return (1.0 + p) * result_narrow - p * result_wide
