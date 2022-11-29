import torch

from helpers.index_helper import IndexHelper


def _actual_noise(pos, i):
    a_b = i.vec2_const(12.9898, 78.233)
    c = 43758.545
    dt = i.dot(pos, a_b)
    sn = torch.fmod(dt, 3.14)
    return torch.frac(torch.sin(sn) * c)


class NoiseEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, noise_shape_texture, noise_scale):
        i = IndexHelper(noise_shape_texture)

        noise_scale = i.view(noise_scale) * i.input_size_factor()
        original_pos = (i.get_p_start() * i.tex_size())
        pos = (original_pos + i.vec2_const(711.0, 911.0)) / noise_scale
        new_pos = torch.floor(pos)
        new_rem = pos - new_pos

        n = _actual_noise(new_pos, i)
        nr = _actual_noise(new_pos + i.vec2_const(1.0, 0.0), i)
        nd = _actual_noise(new_pos + i.vec2_const(0.0, 1.0), i)
        nrd = _actual_noise(new_pos + i.vec2_const(1.0, 1.0), i)

        new_rem_x = i.idx(new_rem, 'x')
        new_rem_y = i.idx(new_rem, 'y')

        h1 = torch.lerp(n, nr, new_rem_x)
        h2 = torch.lerp(nd, nrd, new_rem_x)

        v = torch.lerp(h1, h2, new_rem_y)
        # de morgan, includes initial early out of shader.
        v = torch.where(
            (v >= 0.5).logical_or(i.idx(original_pos, "x") < 1.0).logical_or(i.idx(original_pos, "y") < 1.0).logical_or(i.idx(original_pos, "x") >= (i.tex_size("w") - 1.0)).logical_or(i.idx(original_pos, "y") >= (i.tex_size("h") - 1.0)),
            torch.ones_like(v), torch.zeros_like(v))
        v = v + 2.0 * _actual_noise(pos, i)

        return v
