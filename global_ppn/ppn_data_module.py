import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torchvision.transforms import RandomCrop

from helpers import np_to_torch
from helpers.visual_parameter_def import VisualParameterDef


class PPNDataset(Dataset):
    def __init__(self, root_folder, n_params, crop_dim=None, fixed_params=None, enable_augmentation=False):
        self.img_files = list(sorted(Path(root_folder).glob("*.png"))) + list(sorted(Path(root_folder).glob("*.jpg")))
        self.fixed_params = fixed_params
        self.n_params = n_params

        if crop_dim is not None and enable_augmentation:
            self.transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  RandomCrop(crop_dim)])
        elif enable_augmentation:
            self.transforms = transforms.RandomHorizontalFlip()
        elif crop_dim is not None:
            self.transforms = transforms.CenterCrop(crop_dim)
        else:
            self.transforms = None

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        try:
            img = Image.open(self.img_files[index]).convert("RGB")
        except UnidentifiedImageError:
            print(f"bad image {self.img_files[index]}")
            return None

        if self.transforms is not None:
            img = self.transforms(img)

        img = np_to_torch(img, add_batch_dim=False)
        if self.fixed_params is None:
            result_params = VisualParameterDef.rand((self.n_params, ))
        else:
            result_params = self.fixed_params[index]

        return img, result_params

    @staticmethod
    def get_len_folder(fname):
        return len(list(Path(fname).glob("*.png")) + list(Path(fname).glob("*.jpg")))


class PPNFilterParametersDataModule(pl.LightningDataModule):
    def __init__(self, root_path, n_params=1, crop_dim=None, batch_size=8, num_workers=8,
                 debug_one_item=False):
        super().__init__()
        self.train_set = None
        self.val_set = None
        self.test_set = None

        self.val_fixed_params = None
        self.test_fixed_params = None

        self.batch_size = batch_size
        self.crop_dim = crop_dim
        self.num_workers = num_workers

        self.n_params = n_params

        self.CONFIG = {
            "train_f": f"{root_path}ffhq_train",
            "val_f": f"{root_path}ffhq_val",
            "test_f": f"{root_path}ffhq_val"}

        if debug_one_item:
            self.CONFIG = {
                "train_f": f"{root_path}ffhq_1x50",
                "val_f": f"{root_path}ffhq_1x50",
                "test_f": f"{root_path}ffhq_1x50"}

    def prepare_data(self, *args, **kwargs):
        super().prepare_data()

        if "seed" in kwargs:
            torch.random.manual_seed(kwargs["seed"])

        self.val_fixed_params = VisualParameterDef.rand((PPNDataset.get_len_folder(self.CONFIG["val_f"]), self.n_params))
        self.test_fixed_params = VisualParameterDef.rand((PPNDataset.get_len_folder(self.CONFIG["test_f"]), self.n_params))

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.val_set = PPNDataset(
                self.CONFIG["val_f"],
                self.n_params,
                crop_dim=self.crop_dim,
                fixed_params=self.val_fixed_params)
            self.train_set = PPNDataset(
                self.CONFIG["train_f"],
                self.n_params,
                crop_dim=self.crop_dim,
                enable_augmentation=True)  # fully random parameters

        if stage == 'test' or stage is None:
            self.test_set = PPNDataset(self.CONFIG["test_f"],
                                       self.n_params,
                                       crop_dim=self.crop_dim,
                                       fixed_params=self.test_fixed_params)

    @staticmethod
    def skip_none_collate(batch):
        batch = [x for x in batch if x is not None]

        if len(batch) == 0:
            return batch

        return default_collate(batch)

    @staticmethod
    def get_npr_test_loader(index, effect_name, vpd, num_workers=8, parameter_names=None):
        n_params = len(vpd.vp_ranges)

        path = f"{os.path.dirname(__file__)}/../experiments/nprp/level{index}"
        path_preset = f"{path}/presets_{effect_name}.pt"

        if Path(path_preset).exists():
            test_fixed_presets = torch.load(path_preset)
            test_fixed_presets = vpd.scale_parameters(test_fixed_presets, True)  # scale back
            print("loaded presets")
        else:
            test_fixed_presets = vpd.rand((PPNDataset.get_len_folder(path), n_params))
            torch.save(vpd.scale_parameters(test_fixed_presets), path_preset)
            print("stored presets")

        if parameter_names is not None:
            test_fixed_presets = vpd.select_parameters(test_fixed_presets, parameter_names)

        test_set = PPNDataset(path, n_params, fixed_params=test_fixed_presets)
        return DataLoader(test_set, num_workers=num_workers)

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, num_workers=self.num_workers,  # shuffle=True automatic?
                          collate_fn=self.skip_none_collate)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, num_workers=self.num_workers)
