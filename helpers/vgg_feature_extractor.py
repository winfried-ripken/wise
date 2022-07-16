from collections import namedtuple

import torch
from torch import nn
from torchvision import models


class Vgg19FeatureExtractor(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        self.query_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        # normalize
        X = (X - self.mean.to(X.device)) / self.std.to(X.device)
        self.vgg_pretrained_features.eval()

        i = 0
        results = []
        for layer in self.vgg_pretrained_features:
            X = layer(X)

            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            if name in self.query_layers:
                results.append(X)

        vgg_outputs = namedtuple("VggOutputs", self.query_layers)
        out = vgg_outputs(*results)
        return out
