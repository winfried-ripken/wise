import torch
from torch import nn


class ScaleVisualParameters(nn.Module):
    def __init__(self, vp_ranges):
        super().__init__()

        # register buffers s.t. the device is updated
        self.params_min = torch.zeros((len(vp_ranges),))  # , persistent=False
        self.params_span = torch.zeros((len(vp_ranges),))  # , persistent=False

        i = 0
        for n, low, high in vp_ranges:
            self.params_min[i] = torch.tensor(low)
            self.params_span[i] = torch.tensor(high - low)
            i += 1

    def forward(self, tensor, scale_back=False):
        p_min = self.params_min.type_as(tensor)
        p_span = self.params_span.type_as(tensor)

        if len(tensor.size()) == 4:
            # parameters have spatial dims
            p_min = p_min.view(-1, 1, 1)
            p_span = p_span.view(-1, 1, 1)

        if scale_back:
            return (tensor - p_min) / p_span - 0.5

        return (tensor + 0.5) * p_span + p_min
