import torch

from helpers.index_helper import IndexHelper


def _get_step_size_factor(i, mode):
    if mode is None:
        return i.input_size_factor()
    if mode == 0:
        return 1.0
    elif mode == 1:
        return i.input_size_factor()
    elif mode == 2:
        return math.sqrt(i.input_size_factor())


class FlowAlignedSmoothingEffect(torch.nn.Module):
    def __init__(self, tangent_direction=True):
        super().__init__()

        self.step_size = 0.3333
        self.step_size_scaling_factor = 1.0
        self.tangent_direction = tangent_direction

    def forward(self, x, tangent, sigma, step_size=None, step_size_scaling_factor=None, precisionFactor = 1.0):
        i = IndexHelper(x)

        if step_size is None:
            step_size = torch.tensor(self.step_size, device=x.device)

        sigma = i.view(sigma) * i.input_size_factor() * precisionFactor
        step_size = step_size * _get_step_size_factor(i, step_size_scaling_factor)

        halfWidth = 2.0 * sigma
        twoSigma2 = 2.0 * sigma * sigma
        step = i.view(1.0 / step_size)

        C = x.clone()
        Sum = i.const(1.0)

        def smoothed(c, c_sum, direction):
            if self.tangent_direction:
                v = direction.clone()
            else:
                # grad direction
                v = i.vec2(i.idx(direction.clone(), "y"), -i.idx(direction.clone(), "x"))

            p = (i.get_p_start() + (v / i.tex_size())).clone()
            r = step.clone()

            while torch.any(r < halfWidth):
                k = torch.exp(-r * r / twoSigma2)
                k = torch.where(r < halfWidth, k, torch.zeros_like(k))
                k = i.const(k)
                k = torch.where((i.idx(p, "x") >= 0.0).logical_and(i.idx(p, "x") < 1.0).logical_and(
                    i.idx(p, "y") >= 0.0).logical_and(i.idx(p, "y") < 1.0),
                                k, torch.zeros_like(k))
                c += i.sample(x, p) * k
                c_sum += k

                tf = i.sample(tangent.clone(), p)
                if not self.tangent_direction:
                    # grad direction
                    tf = i.vec2(i.idx(tf, "y"), -i.idx(tf, "x"))

                vt = i.dot(v, tf)
                tf = torch.where(vt < 0.0, -tf, tf)

                v = tf.clone()
                p = (p + tf / i.tex_size()).clone()
                r = (r + step).clone()

        tangent_sampled = tangent if i.same_img_size(x, tangent) else i.sample(tangent, i.get_p_start())
        smoothed(C, Sum, tangent_sampled)
        smoothed(C, Sum, -tangent_sampled)

        return C / Sum
