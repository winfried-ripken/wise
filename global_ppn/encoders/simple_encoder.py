import torch
from torch.nn import Module, Flatten

from helpers.custom_functions import activation_f


class SimpleEncoder(Module):
    def __init__(self, num_classes, nchannels_in=6, activation="linear"):
        super().__init__()
        self.activation = activation

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(nchannels_in, 16, 5, 1, 2),
            # torch.nn.BatchNorm2d(16),  NO Batchnorm!
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 5, 1, 2),
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, 1, 2),
            # torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.AdaptiveMaxPool2d(64)
        )

        self.classifier = torch.nn.Sequential(
            Flatten(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64*64*64, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return activation_f(x, self.activation)
