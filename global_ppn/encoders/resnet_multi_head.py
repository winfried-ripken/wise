import torch
from torch import nn
from torch.hub import load_state_dict_from_url
from torchvision.models import resnet50
from torchvision.models.resnet import model_urls


class ResnetMultiHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        n_features_resnet = 4096
        self.freeze_pretrained_layers = True

        self.res_net = resnet50(num_classes=n_features_resnet, norm_layer=nn.Identity)
        self.load_pretrained_weights()
        del self.res_net.conv1
        self.res_net.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 6 ch. instead of 3

        self.relu = nn.LeakyReLU()

        multi_head = []
        self.num_classes = num_classes
        for i in range(num_classes):
            multi_head.append(nn.Sequential(nn.Linear(n_features_resnet, n_features_resnet // 4),
                                            nn.LeakyReLU(),
                                            nn.Linear(n_features_resnet // 4, n_features_resnet // 4),
                                            nn.LeakyReLU(),
                                            nn.Linear(n_features_resnet // 4, 1)))
        self.multi_head = nn.ModuleList(multi_head)

    def load_pretrained_weights(self):
        model_dict = self.res_net.state_dict()
        state_dict = load_state_dict_from_url(model_urls["resnet50"], progress=True)
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]

        for n in list(state_dict.keys()):
            if "bn" in n or "running" in n or "0.downsample.1" in n:  # remove all batch norm
                del state_dict[n]

        model_dict.update(state_dict)

        self.res_net.load_state_dict(model_dict)
        ppp = self.res_net.named_parameters()

        if self.freeze_pretrained_layers:
            for n, p in ppp:
                if "fc" not in n:
                    # freeze pretrained layers
                    p.requires_grad = False

    def forward(self, x):
        resnet_features = self.res_net(x)
        resnet_features = self.relu(resnet_features)

        result = []
        for i in range(self.num_classes):
            result.append(self.multi_head[i](resnet_features))

        return 0.5 * torch.tanh(torch.cat(result, dim=1))
