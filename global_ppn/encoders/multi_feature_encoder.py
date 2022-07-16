import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d, Sequential, ModuleList, Flatten
from torchvision.models import vgg11

from helpers.losses import gram_matrix


class MultiFeatureEncoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.vgg = vgg11(pretrained=True)
        del self.vgg.classifier

        self.spatial_dim = 920
        self.final_features_gram = 24
        self.final_features_stride = 8
        self.max_spatial_dim = 12
        self.multi_head_dim = 2048

        i = 0
        layer_dims = [self.spatial_dim,
                      self.spatial_dim // 2,
                      self.spatial_dim // 4,
                      self.spatial_dim // 4,
                      self.spatial_dim // 8,
                      self.spatial_dim // 8,
                      self.spatial_dim // 16,
                      self.spatial_dim // 16]

        stride_convs = []
        one_by_one_convs = []
        latent_dim = 0

        for layer in self.vgg.features:
            if isinstance(layer, nn.Conv2d):
                sc = []

                sd = layer_dims[i]
                while sd > self.max_spatial_dim:
                    sc.append(Conv2d(layer.out_channels if len(sc) == 0 else self.final_features_stride,
                                     self.final_features_stride, kernel_size=4, stride=4))
                    sd //= 4

                stride_convs.append(Sequential(*sc))
                one_by_one_convs.append(Conv2d(layer.out_channels, self.final_features_gram, kernel_size=1))
                latent_dim += sd * sd * self.final_features_stride + self.final_features_gram * self.final_features_gram

                i += 1

        self.stride_convs = ModuleList(stride_convs)
        self.one_by_one_convs = ModuleList(one_by_one_convs)
        self.flat = Flatten()

        multi_head = []
        self.num_classes = num_classes
        for i in range(num_classes):
            multi_head.append(nn.Sequential(nn.Linear(latent_dim * 2, self.multi_head_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.multi_head_dim, self.multi_head_dim),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.multi_head_dim, 1)))
        self.multi_head = nn.ModuleList(multi_head)

    def forward(self, tensor):
        tensor = F.interpolate(tensor, (self.spatial_dim, self.spatial_dim))

        x = tensor[:, :3]
        y = tensor[:, 3:]
        idx = 0

        features = []
        for layer in self.vgg.features:
            x = layer(x.clone())
            y = layer(y.clone())

            if isinstance(layer, nn.Conv2d):
                features.append(self.flat(self.stride_convs[idx](x)))
                features.append(self.flat(self.stride_convs[idx](y)))
                features.append(self.flat(gram_matrix(self.one_by_one_convs[idx](x))))
                features.append(self.flat(gram_matrix(self.one_by_one_convs[idx](y))))
                idx += 1

        feature_vector = torch.cat(features, dim=1)

        result = []
        for i in range(self.num_classes):
            result.append(self.multi_head[i](feature_vector))

        return 0.5 * torch.tanh(torch.cat(result, dim=1))


if __name__ == '__main__':
    m = MultiFeatureEncoder(5)
    print(m)
