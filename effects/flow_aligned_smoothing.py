import torch

from helpers.index_helper import IndexHelper


class FlowAlignedSmoothingEffect(torch.nn.Module):
    def __init__(self, tangent_direction=True):
        super().__init__()

        self.step_size = 0.3333
        self.tangent_direction = tangent_direction

    def forward(self, x, tangent, sigma):
        i = IndexHelper(x)

        sigma = i.view(sigma) * i.input_size_factor()
        step_size = self.step_size * i.input_size_factor()
        halfWidth = 2.0 * sigma
        twoSigma2 = 2.0 * sigma * sigma
        step = i.view(torch.tensor(1.0 / step_size, device=x.device))

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

            while torch.sum(r < halfWidth) > 0:
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

        smoothed(C, Sum, tangent)
        smoothed(C, Sum, -tangent)

        return C / Sum
