#
# Source code: https://github.com/wjwjww/OLI2MSI
#

import os
import rasterio
import torch

from torch.utils.data import Dataset, DataLoader

from utils import load_fun


def load_file(filename):
    with rasterio.open(filename) as src:
        file_ = src.read()
    return file_


def _load_files_dir(fdir):
    print('load files from {}'.format(fdir))
    file_list = []
    for dirpath, dirs, files in os.walk(fdir):
        for filename in files:
            if filename.endswith('.TIF'):
                file_list.append(os.path.join(dirpath, filename))
    return sorted(file_list)


class OLI2MSI(Dataset):
    def __init__(self, cfg, is_training=True):
        print('dataset OLI2MSI')
        self.root_path = cfg.dataset.root_path
        self.relname = 'train'
        if not is_training:
            self.relname = 'test'
        self.fdir_lr = os.path.join(self.root_path, self.relname + '_lr')
        self.fdir_hr = os.path.join(self.root_path, self.relname + '_hr')
        self._load_files()

    def _load_files(self):
        self.files_lr = _load_files_dir(self.fdir_lr)
        self.files_hr = _load_files_dir(self.fdir_hr)

    def __len__(self):
        return len(self.files_lr)

    def __getitem__(self, index):
        file_lr = load_file(self.files_lr[index])
        file_hr = load_file(self.files_hr[index])
        return file_lr, file_hr


def load_dataset(cfg, only_test=False, concat_datasets=False):
    collate_fn = cfg.dataset.get('collate_fn')
    if collate_fn is not None:
        collate_fn = load_fun(collate_fn)

    persistent_workers = False
    if cfg.num_workers > 0:
        persistent_workers = True

    train_dset = None
    train_dloader = None
    concat_dloader = None

    if concat_datasets:
        train_dset = OLI2MSI(cfg, is_training=True)
        val_dset = OLI2MSI(cfg, is_training=False)
        dset = torch.utils.data.ConcatDataset([train_dset, val_dset])

        shuffle = True

        concat_dloader = DataLoader(
            dset,
            batch_size=cfg.batch_size,
            sampler=None,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg)
        )

    if not only_test:
        train_dset = OLI2MSI(cfg, is_training=True)

        train_dloader = DataLoader(
            train_dset,
            batch_size=cfg.batch_size,
            sampler=None,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg),
        )

    val_dset = OLI2MSI(cfg, is_training=False)

    shuffle = False

    # TODO distribute also val_dset
    val_dloader = DataLoader(
        val_dset,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn(cfg)
    )

    return train_dloader, val_dloader, concat_dloader
