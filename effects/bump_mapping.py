import math
import torch

from helpers.index_helper import IndexHelper


class BumpMappingEffect(torch.nn.Module):
    @staticmethod
    def forward(texture, sigma_color, bump_scale, phong_shininess, phong_specular, sample_distance):
        i = IndexHelper(texture)
        # x, _ = i.get_rgb_and_alpha(x)  # do not use alpha if present

        sigma_color = sigma_color * i.input_size_factor()
        bump_scale = bump_scale * torch.sqrt(sigma_color) / 5.0

        sample_distance = sample_distance * math.sqrt(i.input_size_factor())

        p = i.get_p_start()
        p = p.unsqueeze(1).repeat(1, 9, 1, 1, 1)

        dx = 1.0 / i.tex_size()[:, 0] * sample_distance
        dy = 1.0 / i.tex_size()[:, 1] * sample_distance

        p[:, 1, 0] -= dx
        p[:, 1, 1] += dy

        p[:, 2, 1] += dy

        p[:, 3, 0] += dx
        p[:, 3, 1] += dy

        p[:, 4, 0] -= dx

        p[:, 5, 0] += dx

        p[:, 6, 0] -= dx
        p[:, 6, 1] -= dy

        p[:, 7, 1] -= dy

        p[:, 8, 0] += dx
        p[:, 8, 1] -= dy

        sample_shape = list(bump_scale.shape)
        sample_shape.insert(1, 1)
        samples = i.sample_complex(texture, p) * bump_scale.view(sample_shape)

        x, a, b, c, d, e, f, g, h = torch.unbind(samples, dim=1)
        n = i.vec3_const(0.0)
        s = i.const(1.0)
        z = i.const(0.0)

        n += i.cross(i.vec3(s, z, x - d), i.vec3(z, s, a - d))
        n += i.cross(i.vec3(-s, z, a - b), i.vec3(z, -s, x - b))
        n += i.cross(i.vec3(z, -s, x - b), i.vec3(s, z, c - b))
        n += i.cross(i.vec3(z, s, c - e), i.vec3(-s, z, x - e))
        n += i.cross(i.vec3(-s, z, x - e), i.vec3(z, -s, h - e))
        n += i.cross(i.vec3(s, z, h - g), i.vec3(z, s, x - g))
        n += i.cross(i.vec3(z, s, x - g), i.vec3(-s, z, f - g))
        n += i.cross(i.vec3(z, -s, f - d), i.vec3(s, z, x - d))

        n_default = i.vec3_const(0.0)
        i.set_idx(n_default, 'z', i.const(1.0))

        n_length = i.get_len(n)
        n_normalized = n / n_length

        n = torch.where(n_length == 0.0, n_default, n_normalized)

        # |vec3(1.0, 1.0, 1.0)| = 1.0 / sqrt(1.0^2 + 1.0^2 + 1.0^2)
        l = torch.full_like(n, 1.0 / math.sqrt(3.0))
        n_dot_l = i.dot(n, l)
        bump = 0.5 + n_dot_l
        specular = i.safe_pow(n_dot_l, phong_shininess) * phong_specular
        bump_result = i.cat(bump, specular)
        return bump_result
