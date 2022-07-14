import torch
from torch.autograd import Function


class OpenGLRound(Function):
    @staticmethod
    def forward(ctx, raw_result):
        # round at the end of the effect pipeline to 8 bit precision
        result = torch.clamp(raw_result, 0.0, 1.0)
        result *= 255.0
        result = torch.round(result).float()
        return result / 255.0

    @staticmethod
    # straight through
    def backward(ctx, grad_output):
        return grad_output


class StraightTroughFloor(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.floor(x)

    @staticmethod
    # straight through
    def backward(ctx, grad_output):
        return grad_output


def activation_f(x, name):
    if name == "linear":
        return x
    elif name == "sigmoid":
        return torch.sigmoid(x)
    else:
        raise ValueError("activation")


opengl_round = OpenGLRound.apply
straight_through_floor = StraightTroughFloor.apply
