import os
import torch

from functools import lru_cache
from torch.utils.data import Dataset, DataLoader

from utils import load_fun


@lru_cache(maxsize=10)
def cached_torch_load(filename):
    return torch.load(filename)


class Sen2VenusDataset(Dataset):
    def __init__(self, cfg, is_training=True):
        self.root_path = cfg.dataset.root_path
        self.relname = 'train'
        if not is_training:
            self.relname = 'test'
        self.fdir = os.path.join(self.root_path, self.relname)
        self._load_files()
        self._filter_files(cfg)

    def _filter_files(self, cfg):
        places = cfg.dataset.get('places')
        if places is not None and not places == []:
            self.files = list(filter(
                lambda name: name.lower().split('_')[1] in places,
                self.files))

    def _load_files(self):
        print('load {} files from {}'.format(self.relname, self.fdir))
        self.files = []
        for dirpath, dirs, files in os.walk(self.fdir):
            for filename in files:
                if filename.endswith('.pt'):
                    self.files.append(os.path.join(dirpath, filename))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        return cached_torch_load(self.files[index])


def load_dataset(cfg, only_test=False, concat_datasets=False):
    dataset_cls = globals()[cfg.dataset.get('cls', 'Sen2VenusDataset')]
    hr_name = cfg.dataset.hr_name
    lr_name = cfg.dataset.lr_name

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
        train_dset = dataset_cls(cfg, is_training=True)
        val_dset = dataset_cls(cfg, is_training=False)
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
            collate_fn=collate_fn(cfg, hr_name=hr_name, lr_name=lr_name)
        )

    if not only_test:
        train_dset = dataset_cls(cfg, is_training=True)

        train_dloader = DataLoader(
            train_dset,
            batch_size=cfg.batch_size,
            sampler=None,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg, hr_name=hr_name, lr_name=lr_name)
        )

    val_dset = dataset_cls(cfg, is_training=False)

    val_dloader = DataLoader(
        val_dset,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn(cfg, hr_name=hr_name, lr_name=lr_name)
    )

    return train_dloader, val_dloader, concat_dloader
