import torch
import torch.nn.functional as F


class UpOrDownSamplingEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, sizes, *param_tensors):
        mode = "bilinear"
        upOrDownSamplingTexture = F.interpolate(x, sizes, mode=mode, align_corners=False)
        param_result = []

        for p in param_tensors:
            # check if local parameter map
            if len(p.shape) == 4:
                param_result.append(F.interpolate(p, sizes, mode=mode, align_corners=False))
            else:
                param_result.append(p)

        # we had to do this because of python 3.7
        if len(param_result) == 1:
            return upOrDownSamplingTexture, param_result[0]
        if len(param_result) == 2:
            return upOrDownSamplingTexture, param_result[0], param_result[1]
        if len(param_result) == 3:
            return upOrDownSamplingTexture, param_result[0], param_result[1], param_result[2]
        else:
            return upOrDownSamplingTexture
