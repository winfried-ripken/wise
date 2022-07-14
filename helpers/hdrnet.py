import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=(0, 1, 0, 1), stride=1, use_bias=True, activation=nn.ReLU,
                 batch_norm=False):
        super(ConvBlock, self).__init__()

        self.padding = padding
        self.conv = nn.Conv2d(int(inc), int(outc), kernel_size, padding=0, stride=stride, bias=use_bias)
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm2d(outc) if batch_norm else None

    def forward(self, x):
        x = F.pad(x, self.padding, "constant", 0.0)
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class HdrNet(nn.Module):
    def __init__(self, nin=4, nout=3):
        super().__init__()
        self.nin = nin
        self.nout = nout

        lb = 8
        cm = 1
        sb = 16
        bn = False
        nsize = 256

        self.luma_bins = lb
        self.channel_multiplier = cm

        self.relu = nn.ReLU()

        # splat features
        n_layers_splat = int(np.log2(nsize / sb))
        self.splat_features = nn.ModuleList()
        prev_ch = 3
        for i in range(n_layers_splat):
            use_bn = bn if i > 0 else False
            self.splat_features.append(ConvBlock(prev_ch, cm * (2 ** i) * lb, 3, stride=2, batch_norm=use_bn))
            prev_ch = splat_ch = cm * (2 ** i) * lb

        # global features
        n_layers_global = int(np.log2(sb / 4))
        print(n_layers_global)
        self.global_features_conv = nn.ModuleList()
        self.global_features_fc = nn.ModuleList()
        for i in range(n_layers_global):
            self.global_features_conv.append(ConvBlock(prev_ch, cm * 8 * lb, 3, stride=2, batch_norm=bn))
            prev_ch = cm * 8 * lb

        n_total = n_layers_splat + n_layers_global
        prev_ch = prev_ch * (nsize / 2 ** n_total) ** 2
        self.global_features_fc.append(FC(prev_ch, 32 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(32 * cm * lb, 16 * cm * lb, batch_norm=bn))
        self.global_features_fc.append(FC(16 * cm * lb, 8 * cm * lb, activation=None, batch_norm=bn))

        # local features
        self.local_features = nn.ModuleList()
        self.local_features.append(ConvBlock(splat_ch, 8 * cm * lb, 3, batch_norm=bn, padding=(1, 1, 1, 1)))
        self.local_features.append(
            ConvBlock(8 * cm * lb, 8 * cm * lb, 3, activation=None, padding=(1, 1, 1, 1), use_bias=False))

        # predicton
        self.conv_out = ConvBlock(8 * cm * lb, lb * nout * nin, 1, padding=(0, 0, 0, 0), activation=None)

    def forward(self, lowres_input):
        bs = lowres_input.shape[0]
        lb = self.luma_bins
        cm = self.channel_multiplier

        x = lowres_input
        for layer in self.splat_features:
            x = layer(x)

        splat_features = x

        for layer in self.global_features_conv:
            x = layer(x)

        x = x.reshape(bs, -1)
        for layer in self.global_features_fc:
            x = layer(x)
        global_features = x

        x = splat_features
        for layer in self.local_features:
            x = layer(x)
        local_features = x

        fusion_grid = local_features
        fusion_global = global_features.reshape(bs, 8 * cm * lb, 1, 1)
        fusion = self.relu(fusion_grid + fusion_global)

        x = self.conv_out(fusion)

        x = torch.stack(torch.split(x, 8, 1), 1)
        x = torch.stack(torch.split(x, 3, 1), 1).permute(0, 2, 1, 3, 4, 5)

        return x
