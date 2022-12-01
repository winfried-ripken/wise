import torch

from helpers.index_helper import IndexHelper


class LicT:
    def __init__(self, p, t, i):
        self.p = p
        self.t = t
        self.i = i
        self.w = i.const(0.0)
        self.dw = i.const(0)


class XDoGPass1Effect(torch.nn.Module):
    def __init__(self, invert_thresholding=False):
        super().__init__()
        self.invert_thresholding = invert_thresholding

    @staticmethod
    def my_step(tangent, s):
        i = s.i

        t = i.idx(i.sample(tangent, s.p), "xy")  # sample only if s.w is within range?
        t = torch.where(i.dot(t, s.t) < 0.0, -t, t)
        s.t = t.clone()

        res = torch.abs(i.safe_div((torch.frac(s.p) - 0.5 - torch.sign(t)), t))
        s.dw = torch.where(torch.abs(i.idx(t, "x")) > torch.abs(i.idx(t, "y")),
                           i.idx(res, "x"), i.idx(res, "y"))

        s.p = (s.p + t * s.dw / i.tex_size()).clone()
        s.w = (s.w + s.dw).clone()

    def forward(self, x, tangent, epsilon, sigma_edge, phi):
        i = IndexHelper(x)
        epsilon = i.view(epsilon)

        sigmaEdge = i.view(sigma_edge) * i.input_size_factor()
        twoSigmaEdgeSquare = 2.0 * sigmaEdge * sigmaEdge
        halfStepWidth = 2.0 * sigmaEdge

        p = i.get_p_start()
        a = LicT(p.clone(), i.idx(tangent, "xy").clone(), i)
        b = LicT(p.clone(), -i.idx(tangent, "xy").clone(), i)

        W = i.const(1.0)
        H = i.idx(x, "x")

        def iterate(s, h, w):
            while torch.sum(s.w.mean(dim=(1, 2, 3), keepdim=True) < halfStepWidth * 0.9) > 0:
                sw_old = s.w.clone()

                self.my_step(tangent, s)
                k = s.dw * torch.exp(-s.w * s.w / (twoSigmaEdgeSquare + torch.finfo(twoSigmaEdgeSquare.dtype).eps))
                k = torch.where((i.idx(s.p, "x") >= 0.0).logical_and(i.idx(s.p, "x") < 1.0)
                                .logical_and(i.idx(s.p, "y") >= 0.0).logical_and(i.idx(s.p, "y") < 1.0)
                                .logical_and(sw_old < halfStepWidth), k, torch.zeros_like(k))

                h += k * i.idx(i.sample(x, s.p), "x")
                w += k

        iterate(a, H, W)
        iterate(b, H, W)

        H /= W

        if self.invert_thresholding:
            H = 1.0 - H

        H = torch.where(H > epsilon,
                        torch.ones_like(H),
                        1 + torch.tanh(100.0 * i.view(phi) * (H - epsilon)))

        return torch.cat([i.idx(1 - H, "x"), torch.zeros_like(i.idx(1 - H, "x")), torch.zeros_like(i.idx(1 - H, "x"))], dim=1)
