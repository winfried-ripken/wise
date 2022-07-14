import torch


class IdentityEffect(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args):
        return x
