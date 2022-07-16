import torch

from helpers.index_helper import IndexHelper


class StructureTensorEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sigma, scale_luminance=True):
        i = IndexHelper(x)
        x, _ = i.get_rgb_and_alpha(x)  # do not use alpha if present

        p = i.get_p_start()
        p = p.unsqueeze(1).repeat(1, 8, 1, 1, 1)
        sigma = i.view(sigma, dims=3)

        dx = sigma * i.input_size_factor() / i.tex_size()[:, 0]
        dy = sigma * i.input_size_factor() / i.tex_size()[:, 1]

        p[:, 0, 0] -= dx
        p[:, 0, 1] -= dy

        p[:, 1, 0] -= dx

        p[:, 2, 0] -= dx
        p[:, 2, 1] += dy

        p[:, 3, 1] -= dy

        p[:, 4, 1] += dy

        p[:, 5, 0] += dx
        p[:, 5, 1] -= dy

        p[:, 6, 0] += dx

        p[:, 7, 0] += dx
        p[:, 7, 1] += dy

        s = i.sample_complex(x, p)

        kernel_u = torch.tensor([-0.25, -0.5, -0.25, 0, 0, 0.25, 0.5, 0.25], device=x.device).reshape(1, 8, 1, 1, 1)
        kernel_v = torch.tensor([-0.25, 0, 0.25, -0.5, 0.5, -0.25, 0, 0.25], device=x.device).reshape(1, 8, 1, 1, 1)

        s_u = (s * kernel_u).sum(dim=1)
        s_v = (s * kernel_v).sum(dim=1)

        if scale_luminance:
            s_u *= i.vec3_const(100.0, 1.0, 1.0)
            s_v *= i.vec3_const(100.0, 1.0, 1.0)

        return i.vec3(i.dot(s_u, s_u), i.dot(s_v, s_v), i.dot(s_u, s_v))
