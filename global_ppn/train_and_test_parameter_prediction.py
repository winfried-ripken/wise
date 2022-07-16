import os
import sys

from effects import xdog_params

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from pytorch_lightning.loggers import TensorBoardLogger
from effects.xdog import XDoGEffect

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# from effects.toon import ToonEffect
from helpers import torch_to_np
from helpers.visual_parameter_def import portrait_preset
from global_ppn.parameter_prediction_net import DirectStyleReverseEngineeringNet, \
    GradientStyleReverseEngineeringNet, AggregateTestMetrics
from global_ppn.ppn_data_module import PPNFilterParametersDataModule


CONFIG = {
    "load_path": f"{os.path.dirname(__file__)}/../trained_models/global/ppn_xdog.ckpt",  # only needed if training should be resumed
    "crop_dim": (920, 920),  # (920, 920), None
    "ppn_type": GradientStyleReverseEngineeringNet,  # GradientStyleReverseEngineeringNet
    "effect": "xdog",
    "enable_checkpoints": True,
    "lr": 1e-5,
    "max_epochs": 25,
    "accumulate_grad_batch_size": 64,  # add some stability by using a larger virtual batch size
    "num_workers": 2,

    # TESTING
    "ppn_type_test": AggregateTestMetrics,
    "skip_images": 0,
    "test_level_display": 2}


def train(loss, architecture, load, bs, gpus, num_workers, root_path, trainable_parameters, default_preset, effect_type,
          debug, log_dir):
    load_path = CONFIG["load_path"]

    print(f"training parameters: {trainable_parameters}")

    data_module = PPNFilterParametersDataModule(n_params=len(trainable_parameters),
                                                batch_size=bs, crop_dim=CONFIG["crop_dim"],
                                                num_workers=num_workers,
                                                debug_one_item=debug,
                                                root_path=root_path)

    ppn = CONFIG["ppn_type"](parameter_names=trainable_parameters,
                             effect_type=effect_type, loss_name=loss,
                             enable_checkpoints=CONFIG["enable_checkpoints"],
                             encoder_architecture=architecture, lr=CONFIG["lr"],
                             default_preset=default_preset)  # debug_grads=True

    tb_logger = TensorBoardLogger(log_dir, default_hp_metric=False, name=effect_type.__name__)

    quantity = 'val_loss' if CONFIG["ppn_type"] == GradientStyleReverseEngineeringNet else "parameter_loss"
    save_best = ModelCheckpoint(tb_logger.log_dir, monitor=quantity, save_last=True, save_top_k=1)  # dirpath=save_dir
    lr_monitor = LearningRateMonitor()

    acc_batch = CONFIG["accumulate_grad_batch_size"] // bs
    print(f"accumulate {acc_batch} batches. Effective batch size is: {acc_batch * bs}")

    acc = "dp"
    if len(gpus) > 1:
        acc = "ddp"

    trainer = pl.Trainer(callbacks=[save_best, lr_monitor], gpus=gpus, accelerator=acc,
                         resume_from_checkpoint=load_path if load else None,
                         accumulate_grad_batches=acc_batch,
                         max_epochs=CONFIG["max_epochs"],
                         track_grad_norm=2, default_root_dir=log_dir, logger=tb_logger)
    # , precision=CONFIG["precision"], amp_level=CONFIG["amp_level"]

    trainer.logger.experiment.add_text("training params", f"loss: {loss}\n"
                                                          f"folder: {Path(trainer.logger.log_dir).name}\n"
                                                          f"architecture: {architecture}\n"
                                                          f"trainable params: {trainable_parameters}\n"
                                                          f"effect type: {effect_type}\n"
                                                          f"ppn type: {CONFIG['ppn_type']}")

    print("----------")
    print(f"loss: {loss} - logging to: {trainer.logger.log_dir}")

    trainer.fit(ppn, data_module)


def plot_dataloader(root_path, trainable_parameters, default_preset, effect_type, debug):
    data_module = PPNFilterParametersDataModule(n_params=len(trainable_parameters),
                                                batch_size=1, crop_dim=CONFIG["crop_dim"],
                                                num_workers=0,
                                                debug_one_item=debug,
                                                root_path=root_path)
    data_module.prepare_data()
    data_module.setup()

    effect = effect_type().cuda()
    for p, param in data_module.train_dataloader():
        vp = effect.vpd.preset_tensor(default_preset, p.new_zeros((1,))).cuda()
        vp = effect.vpd.update_visual_parameters(vp, trainable_parameters, param.cuda())

        result = effect(p.cuda(), vp.cuda())
        _, ax = plt.subplots(2)
        ax[0].imshow(torch_to_np(p))
        ax[1].imshow(torch_to_np(result))
        plt.show()


def show_img_compare(axn, x, target_style_image, style_image):
    axn[0].axis('off')
    axn[1].axis('off')
    axn[2].axis('off')

    axn[0].set_title("Source image")
    axn[1].set_title("Ground truth")
    axn[2].set_title("Predicted style")

    axn[0].imshow(torch_to_np(x))
    axn[1].imshow(torch_to_np(target_style_image))
    axn[2].imshow(torch_to_np(style_image))


def load_test_dataset(ppn, root_path, debug, test_npr):
    if test_npr:
        # NPRP Test Set
        dl = [PPNFilterParametersDataModule.get_npr_test_loader(1, type(ppn.effect).__name__,
                                                                ppn.effect.vpd,
                                                                num_workers=CONFIG["num_workers"],
                                                                parameter_names=ppn.parameter_names),
              PPNFilterParametersDataModule.get_npr_test_loader(2, type(ppn.effect).__name__,
                                                                ppn.effect.vpd,
                                                                num_workers=CONFIG["num_workers"],
                                                                parameter_names=ppn.parameter_names),
              PPNFilterParametersDataModule.get_npr_test_loader(3, type(ppn.effect).__name__,
                                                                ppn.effect.vpd,
                                                                num_workers=CONFIG["num_workers"],
                                                                parameter_names=ppn.parameter_names)]
    else:
        # OUR Test Set
        dm = PPNFilterParametersDataModule(n_params=len(ppn.parameter_names), batch_size=1, crop_dim=CONFIG["crop_dim"],
                                           debug_one_item=debug, num_workers=CONFIG["num_workers"],
                                           root_path=root_path)
        dm.prepare_data(seed=1111)
        dm.setup()
        dl = [dm.test_dataloader()]

    return dl


def predict(root_path, path, test_on_cpu, debug, test_scores, test_npr):
    ppn_type = CONFIG["ppn_type_test"]

    if test_on_cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    ppn = ppn_type.load_from_checkpoint(path, strict=False).to(device)
    ppn.eval()
    dl = load_test_dataset(ppn, root_path, debug, test_npr)

    skip = CONFIG["skip_images"]
    n = 2
    fig, ax = plt.subplots(n, 3, figsize=(10, 10))

    # Test scores only
    if test_scores:
        trainer = pl.Trainer(gpus=1)
        trainer.test(ppn, test_dataloaders=dl)
        return None

    print(f"Param & GT & Predicted \\")
    with torch.no_grad():
        for x_batch, target_embedding_batch in dl[CONFIG["test_level_display"] - 1]:
            if n <= 0:
                break

            if skip > 0:
                skip -= 1
                continue

            for i in range(x_batch.size(0)):
                x = x_batch[i:i+1].to(device)
                target_embedding = target_embedding_batch[i:i+1].to(device)

                if n <= 0:
                    break

                target_style_image = ppn.stylize(x, target_embedding)
                embedding = ppn.internal_forward(x, target_style_image)
                style_image = ppn.stylize(x, embedding)

                for j, name in reversed(list(enumerate(ppn.parameter_names))):
                    tt = target_embedding[:, j].squeeze()
                    ss = embedding[:, j].squeeze()
                    print(f"{name} {n} & {tt:.2f} & "
                          fr"{ss:.2f} diff {torch.abs(tt-ss)*100:.2f}% \\")

                n -= 1
                print("\n")

                show_img_compare(ax[n], x, target_style_image, style_image)

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', help='which loss to use', default="l1")
    parser.add_argument('--architecture', help='which architecture to use', default="multi_feature")
    parser.add_argument('--load', dest='load', action='store_true')
    parser.add_argument('--task', default="train")
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--root_path', default=".")
    parser.add_argument('--effect', default="xdog")
    parser.add_argument('--gpus', nargs="*", type=int, default=[0])
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--test_path', default=f"{os.path.dirname(__file__)}/../trained_models/global/ppn_xdog.ckpt")
    parser.add_argument('--test_on_cpu', action='store_true')
    parser.add_argument('--debug_mode', default="run")
    parser.add_argument('--test_scores', action='store_true')    
    parser.add_argument('--log_dir', default="lightning_logs")
    parser.add_argument('--test_npr', action="store_true")
    parser.set_defaults(load=False, test_on_cpu=False, test_scores=False,
                        test_npr=True)

    args = parser.parse_args()
    print(vars(args))

    trainable_parameters = None
    default_preset = None
    effect_type = None
    debug = False

    if args.debug_mode == "debug":
        debug = True

    if args.effect == "xdog":
        trainable_parameters = xdog_params
        default_preset = portrait_preset
        effect_type = XDoGEffect
    # elif args.effect == "toon":
    #    trainable_parameters = toon_params
    #    default_preset = portrait_preset
    #    effect_type = ToonEffect

    if args.task == "train":
        train(args.loss, args.architecture, args.load, args.bs, args.gpus,
              args.num_workers, args.root_path, trainable_parameters, default_preset, effect_type, debug, args.log_dir)
    elif args.task == "predict":
        predict(args.root_path, args.test_path, args.test_on_cpu, debug, args.test_scores, args.test_npr)
    elif args.task == "test_dataloader":
        plot_dataloader(args.root_path, trainable_parameters, default_preset, effect_type, debug)
    else:
        print("nothing to do")


if __name__ == '__main__':
    main()
