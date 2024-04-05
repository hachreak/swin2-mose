import torch

from torch.utils.data import Dataset, DataLoader
from torchgeo.datasets import SeasoNet

from utils import load_fun


class SeasoNetDataset(Dataset):

    classes = [
        'Continuous urban fabric',
        'Discontinuous urban fabric',
        'Industrial or commercial units',
        'Road and rail networks and associated land',
        'Port areas',
        'Airports',
        'Mineral extraction sites',
        'Dump sites',
        'Construction sites',
        'Green urban areas',
        'Sport and leisure facilities',
        'Non-irrigated arable land',
        'Vineyards',
        'Fruit trees and berry plantations',
        'Pastures',
        'Broad-leaved forest',
        'Coniferous forest',
        'Mixed forest',
        'Natural grasslands',
        'Moors and heathland',
        'Transitional woodland/shrub',
        'Beaches, dunes, sands',
        'Bare rock',
        'Sparsely vegetated areas',
        'Inland marshes',
        'Peat bogs',
        'Salt marshes',
        'Intertidal flats',
        'Water courses',
        'Water bodies',
        'Coastal lagoons',
        'Estuaries',
        'Sea and ocean',
    ]

    def __init__(self, cfg, is_training=True):
        print('dataset SEASONET')
        self.root_path = cfg.dataset.root_path
        self.relname = 'train'
        if not is_training:
            self.relname = 'test'

        self.dset = SeasoNet(
            root=self.root_path,
            split=self.relname,
            **cfg.dataset.kwargs
        )

        cfg.classes = self.classes
        self.kwargs = cfg.dataset.kwargs

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        item = self.dset[index]

        if self.kwargs.get('bands') == ['20m']:
            # get only B5, B6, B7, B8A because sen2venus
            item['image'] = item['image'][:4]

        return item


def load_dataset(cfg, only_test=False, concat_datasets=False):
    dataset_cls = load_fun(cfg.dataset.get('cls', 'SeasoNetDataset'))

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
        concat_dloader = DataLoader(
            dset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg),
        )

    if not only_test:
        train_dset = dataset_cls(cfg, is_training=True)
        train_dloader = DataLoader(
            train_dset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn(cfg),
        )

    val_dset = dataset_cls(cfg, is_training=False)
    val_dloader = DataLoader(
        val_dset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn(cfg),
    )

    return train_dloader, val_dloader, concat_dloader
