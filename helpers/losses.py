from functools import reduce

import torch
from PIL import Image
from torch import nn
from torch.nn import MSELoss, L1Loss
from torch.nn.functional import mse_loss, l1_loss
from torchvision.transforms import ToTensor

from effects.rgb_to_lab import RGBToLabEffect
from effects.structure_tensor import StructureTensorEffect
from helpers.color_conversion import rgb_to_yuv
from helpers.hist_layers import SingleDimHistLayer
from helpers.hist_metrics import DeepHistLoss, EarthMoversDistanceLoss
from helpers.index_helper import IndexHelper
from helpers.ms_ssim import MixLoss
from helpers.vgg_feature_extractor import Vgg19FeatureExtractor


def gram_matrix(input):
    a, b, c, d = input.size()  # keep batch dim separate!

    features = input.view(a, b, c * d)
    G = torch.matmul(features, features.transpose(1, 2))

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(b * c * d)


def get_individual_syle_weight(gram_matrices):
    return 3 / (sum(torch.linalg.norm(style_gram_matrix) for style_gram_matrix in gram_matrices) / len(gram_matrices))


class PerceptualLoss(nn.Module):
    def __init__(self, style_image_path, image_dim, style_img_keep_aspect_ratio, style_weight=1e10, content_weight=1e5,
                 lightning_module=None,
                 **kwargs):
        super().__init__()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("style_image_path", "style_weight", "content_weight")

        self.vgg = Vgg19FeatureExtractor()
        self.mse_loss = MSELoss()

        self.content_weight = content_weight
        self.style_weight = style_weight

        style_image = Image.open(style_image_path).convert("RGB")

        style_image = ToTensor()(style_image)
        if image_dim:
            size = image_dim if style_img_keep_aspect_ratio else (image_dim, image_dim)
            style_image = style_image.resize(size)
        self.target_styles = self.generate_targets(style_image.unsqueeze(0))

    def generate_targets(self, style_image):
        features = self.vgg(style_image)
        gram_matrices = [gram_matrix(f) for f in features]
        return [style_gram_matrix * get_individual_syle_weight(gram_matrices)
                for style_gram_matrix in gram_matrices]

    def move_tensors(self, device):
        self.vgg.mean = self.vgg.mean.to(device)
        self.vgg.std = self.vgg.std.to(device)
        self.target_styles = [gram.to(device) for gram in self.target_styles]

    def forward(self, x, y):
        y_feat = self.vgg(y)
        x_feat = self.vgg(x)
        content_loss = self.content_weight * self.mse_loss(x_feat.conv_4, y_feat.conv_4)

        y_grams = [gram_matrix(f) for f in y_feat]
        style_loss = self.style_weight * torch.stack(
            [self.mse_loss(y_gram, target_style.to(x.device).repeat(y_gram.size(0), 1, 1))
             for y_gram, target_style in zip(y_grams, self.target_styles)]).sum()

        return content_loss, style_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--style_weight', type=float, default=1e10)
        parser.add_argument('--content_weight', type=float, default=1e5)
        parser.add_argument('--style_image_path', default="manga_style.png")
        return parent_parser


class PerceptualStyleLoss(nn.Module):
    def __init__(self, style_image_path, image_dim=1024, style_weight=1.0, lightning_module=None, **kwargs):
        super().__init__()

        style_image = Image.open(style_image_path).convert("RGB").resize((image_dim, image_dim))
        style_image = ToTensor()(style_image).unsqueeze(0)

        self.vgg = Vgg19FeatureExtractor()
        self.style_weight = style_weight
        self.target_style = self.generate_targets(style_image)
        self.mse_loss = MSELoss()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("style_weight", "style_image_path")

    def generate_targets(self, style_image):
        features = self.vgg(style_image)
        gram_matrices = [gram_matrix(f) for f in features]
        return [style_gram_matrix * get_individual_syle_weight(gram_matrices)
                for style_gram_matrix in gram_matrices]

    def forward(self, y):
        assert self.target_style is not None
        y_feat = self.vgg(y)
        y_grams = [gram_matrix(f) for f in y_feat]

        style_loss = self.style_weight * torch.stack(
            [self.mse_loss(e[0], e[1].to(e[0].device).repeat(e[0].size(0), 1, 1))
             for e in zip(y_grams, self.target_style)]).sum()

        return style_loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--style_weight', type=float, default=1e5)
        return parent_parser


class PerceptualContentLoss(nn.Module):
    def __init__(self, content_weight=1.0, lightning_module=None, **kwargs):
        super().__init__()
        self.content_weight = content_weight
        self.mse_loss = MSELoss()
        self.vgg = Vgg19FeatureExtractor()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("content_weight")

    def forward(self, x, y):
        y_feat = self.vgg(y)
        x_feat = self.vgg(x)

        return self.content_weight * self.mse_loss(x_feat.conv_4, y_feat.conv_4)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--content_weight', type=float, default=1)
        return parent_parser


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_to_lab = RGBToLabEffect()
        self.structure_tensor = StructureTensorEffect()

    def forward(self, x, y):
        sigma = torch.tensor(0.43137, device=x.device)
        sst_x = torch.clamp(self.structure_tensor(self.rgb_to_lab(x), sigma), -5, 5)
        sst_y = torch.clamp(self.structure_tensor(self.rgb_to_lab(y), sigma), -5, 5)

        return mse_loss(sst_x, sst_y)


# Loss from dehazing paper
class DehazingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_loss = GradientLoss()
        self.content_loss = PerceptualContentLoss()

        for param in self.parameters():
            param.requires_grad = False

        self.lambda_l2 = 1.0
        self.lambda_g = 0.5
        self.lambda_f = 0.8

    def forward(self, x, y):
        l2 = mse_loss(x, y)
        gl = self.grad_loss(x, y)
        cl = self.content_loss(x, y)

        return self.lambda_l2 * l2 + self.lambda_g * gl + self.lambda_f * cl


class TotalVariationLoss(nn.Module):
    def __init__(self, regularizer_weight=1.0, lightning_module=None, **kwargs):
        super().__init__()

        if lightning_module is not None:
            lightning_module.save_hyperparameters("regularizer_weight")

        self.regularizer_weight = regularizer_weight

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Perceptual Loss")
        parser.add_argument('--regularizer_weight', type=float, default=3.0)
        return parent_parser

    def forward(self, x):
        return self.regularizer_weight * (
            torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) +
            torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        )


# This seems to work well
class DeepHistL1Loss(nn.Module):
    # i.e. histogram (color part) + l1 loss
    def __init__(self, emd_factor=0.2, l1_factor=1.0, l2_factor=0.0):
        super().__init__()
        self.single_dim_hist_layer = SingleDimHistLayer()
        self.emd = EarthMoversDistanceLoss()
        self.l1_loss = L1Loss()
        self.l2_loss = MSELoss()
        self.tv_loss = TotalVariationLoss()

        self.emd_factor = emd_factor
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor

    def forward(self, output, target):
        # x and y are RGB images
        assert output.size(1) == 3
        assert target.size(1) == 3

        # this does not work
        # x = resize_keep_aspect_ratio(x)
        # y = resize_keep_aspect_ratio(y)

        x = rgb_to_yuv(output)
        y = rgb_to_yuv(target)

        hist_x = [self.single_dim_hist_layer(x[:, i]) for i in range(3)]
        hist_y = [self.single_dim_hist_layer(y[:, i]) for i in range(3)]

        emd_loss = reduce(torch.add, [self.emd(hist_x[i], hist_y[i]) for i in range(3)]) / 3.0
        return self.emd_factor * emd_loss.mean() + self.l1_factor * self.l1_loss(output, target) \
               + self.l2_factor * self.l2_loss(output, target)  # + 0.0 * self.tv_loss(output)


def l1_loss_ignore_minus_one(prediction, target):
    return torch.where(target >= -0.5, l1_loss(prediction, target), torch.zeros_like(target))


def categorical_proxy_loss(prediction, target):
    ce_loss = torch.nn.CrossEntropyLoss()
    target = torch.floor((target + 0.5) * prediction.size(2)).to(torch.long)
    loss = torch.tensor(0.0, device=prediction.device)

    for i in range(prediction.size(1)):
        loss += ce_loss(prediction[:, i], target[:, i])

    return loss / prediction.size(1)


def loss_from_string(loss):
    if loss == "dehazing":
        loss_f = DehazingLoss()
    elif loss == "perceptual_content":
        loss_f = PerceptualContentLoss()
    elif loss == "mix":
        loss_f = MixLoss()
    elif loss == "histogram":
        loss_f = DeepHistLoss()
    elif loss == "histogram_l1":
        loss_f = DeepHistL1Loss()
    elif loss == "l2":
        loss_f = mse_loss
    elif loss == "l1":
        loss_f = l1_loss
    elif loss == "l1_loss_ignore_minus_one":
        loss_f = l1_loss_ignore_minus_one
    elif loss == "categorical_proxy":
        loss_f = categorical_proxy_loss
    else:
        raise ValueError(f"{loss} is invalid loss")

    return loss_f
