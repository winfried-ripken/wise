import pytorch_lightning as pl
import torch

from torch.nn.functional import mse_loss, l1_loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from global_ppn.encoders.multi_feature_encoder import MultiFeatureEncoder
from global_ppn.encoders.resnet_multi_head import ResnetMultiHead
from global_ppn.encoders.simple_encoder import SimpleEncoder
from helpers.losses import loss_from_string
from helpers.metrics import get_psnr_pt, get_ssim_pt


class ParameterPredictionBase(pl.LightningModule):
    def __init__(self,
                 parameter_names,
                 default_preset,
                 effect_type,
                 lr=1e-4,
                 **kwargs):
        super().__init__()

        self.save_hyperparameters('effect_type', 'parameter_names', 'lr', 'default_preset')
        self.lr = lr
        self.effect = effect_type()

        self.default_preset = default_preset
        self.parameter_names = parameter_names
        self.n_classes = len(parameter_names)

    def configure_optimizers(self):
        filtered_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.Adam(filtered_params, lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "style_image_loss"}


class StyleReverseEngineeringBase(ParameterPredictionBase):
    def __init__(self,
                 activation="linear",
                 encoder_architecture="simple",
                 **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters('activation', 'encoder_architecture')

        # construct encoder
        if encoder_architecture == "mobilenet":
            raise ValueError("mobilenet is deprecated")
        elif encoder_architecture == "simple":
            self.encoder = SimpleEncoder(num_classes=self.n_classes, activation=activation)
        elif encoder_architecture == "multi_feature":
            self.encoder = MultiFeatureEncoder(num_classes=self.n_classes)
        elif encoder_architecture == "resnet":
            self.encoder = ResnetMultiHead(num_classes=self.n_classes)
        else:
            raise ValueError("encoder_architecture")

    def update_default_with_embedding(self, embedding):
        default = self.effect.vpd.preset_tensor(self.default_preset, embedding)
        return self.effect.vpd.update_visual_parameters(default, self.parameter_names, embedding)

    def stylize(self, x, embedding):
        complete_embedding = self.update_default_with_embedding(embedding)
        return self.effect(x, complete_embedding)

    def internal_forward(self, x, x_stylized):
        # return only our partial embedding
        encoder_in = torch.cat([x, x_stylized], dim=1)
        return self.encoder(encoder_in)

    def forward(self, x, x_stylized):
        # return always the full parameter set
        embedding = self.internal_forward(x, x_stylized)
        return self.update_default_with_embedding(embedding)

    def compute_all(self, batch):
        x, target_embedding = batch

        with torch.no_grad():
            target_style_image = self.stylize(x, target_embedding)

        embedding = self.internal_forward(x, target_style_image)
        style_image = self.stylize(x, embedding)

        return style_image, target_style_image, embedding, target_embedding

    def log_metrics(self, style_image, target_style_image, embedding, target_embedding):
        self.log('style_image_loss', mse_loss(style_image, target_style_image), sync_dist=True)
        self.log('parameter_loss', torch.mean(torch.abs(embedding - target_embedding)), sync_dist=True)
        self.log('parameter_loss_l2', mse_loss(embedding, target_embedding), sync_dist=True)
        self.log('ssim', get_ssim_pt(style_image, target_style_image), sync_dist=True)
        self.log('psnr', get_psnr_pt(style_image, target_style_image), sync_dist=True)


class GradientStyleReverseEngineeringNet(StyleReverseEngineeringBase):
    def __init__(self,
                 loss_name,
                 enable_checkpoints=True,
                 debug_grads=True,
                 gamma_l2=0.01,
                 clip_grads=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.save_hyperparameters('enable_checkpoints', 'clip_grads', 'gamma_l2', 'loss_name', 'debug_grads')
        self.debug_grads = debug_grads
        self.gamma_l2 = gamma_l2
        self.clip_grads = clip_grads
        self.loss_f = loss_from_string(loss_name)
        self.acc_grads = []

        if enable_checkpoints:
            self.effect.enable_checkpoints()  # save memory

    def grad_hook(self, grads):
        if self.clip_grads:
            grads.clamp_(-1, 1)  # perform gradient clipping

        grads = grads.detach().sum(dim=0)
        self.acc_grads.append(grads)

    def training_step(self, batch, batch_index):
        if len(batch) == 0:
            # this means an invalid image was passed
            return

        style_image, target_style_image, embedding, target_embedding = self.compute_all(batch)

        if self.debug_grads:
            embedding.register_hook(self.grad_hook)

        loss = self.loss_f(style_image, target_style_image)  # compute loss in image space
        self.log('train_image_loss', loss, sync_dist=True)

        if len(self.acc_grads) > 0:
            for i, p in enumerate(self.parameter_names):
                self.log(f'grad_{p}', torch.mean(torch.stack([grads[i] for grads in self.acc_grads], dim=0), dim=0))

        return loss

    def validation_step(self, batch, batch_index):
        style_image, target_style_image, embedding, target_embedding = self.compute_all(batch)
        self.log_metrics(style_image, target_style_image, embedding, target_embedding)
        self.log('val_loss', self.loss_f(style_image, target_style_image), sync_dist=True)

        for i, p in enumerate(self.parameter_names):
            self.log(f"l1 {p}", torch.mean(torch.abs(embedding[:, i] - target_embedding[:, i])), sync_dist=True)

    def test_step(self, batch, batch_index, dataloader_index=0):
        style_image, target_style_image, embedding, target_embedding = self.compute_all(batch)
        self.log_metrics(style_image, target_style_image, embedding, target_embedding)
        self.log('test_loss', self.loss_f(style_image, target_style_image), sync_dist=True)


class DirectStyleReverseEngineeringNet(StyleReverseEngineeringBase):
    def __init__(self, loss_name, **kwargs):
        self.loss_f = loss_from_string(loss_name)
        self.save_hyperparameters('loss_name')

        super().__init__(**kwargs)

    def training_step(self, batch, batch_index):
        x, target_embedding = batch

        with torch.no_grad():
            target_style_image = self.stylize(x, target_embedding)

        embedding = self.internal_forward(x, target_style_image)
        loss = self.loss_f(embedding, target_embedding)

        self.log('train_loss', loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index):
        style_image, target_style_image, embedding, target_embedding = self.compute_all(batch)
        self.log_metrics(style_image, target_style_image, embedding, target_embedding)

        for i, p in enumerate(self.parameter_names):
            self.log(f"l1 {p}", torch.mean(torch.abs(embedding[:, i] - target_embedding[:, i])), sync_dist=True)

    def test_step(self, batch, batch_index, dataloader_index=0):
        self.validation_step(batch, batch_index)


class AggregateTestMetrics(StyleReverseEngineeringBase):
    def __init__(self, loss_name, **kwargs):
        self.loss_f = loss_from_string(loss_name)
        self.save_hyperparameters('loss_name')

        super().__init__(**kwargs)

    def test_step(self, batch, batch_index, dataloader_index=0):
        style_image, target_style_image, embedding, target_embedding = self.compute_all(batch)
        ssim = get_ssim_pt(style_image, target_style_image)
        psnr = get_psnr_pt(style_image, target_style_image)
        param_loss = l1_loss(embedding, target_embedding)

        self.log('parameter_loss_l1', param_loss, sync_dist=True)
        self.log('ssim', ssim, sync_dist=True)
        self.log('psnr', psnr, sync_dist=True)
        return ssim, psnr, param_loss

    def test_epoch_end(self, outputs):
        res_ssim = []
        res_psnr = []
        res_ploss = []

        for t in outputs:
            for tt in t:
                res_ssim.append(tt[0])
                res_psnr.append(tt[1])
                res_ploss.append(tt[2])

        final_ssim = torch.stack(res_ssim).mean()
        final_psnr = torch.stack(res_psnr).mean()
        final_ploss = torch.stack(res_ploss).mean()

        self.log('final_ssim', final_ssim, sync_dist=True)
        self.log('final_psnr', final_psnr, sync_dist=True)
        self.log('final_ploss', final_ploss, sync_dist=True)

        return final_ssim, final_psnr, final_ploss
