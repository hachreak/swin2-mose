import torch

from torchvision import transforms
from torch.utils.data import default_collate
from tqdm import tqdm

from .v3 import MeanStd


def collate_fn(cfg):
    print('collate_fn')

    def do_nothing(x):
        return x
    norm_fun = do_nothing

    if 'stats' in cfg.dataset and cfg.phase != 'plot_data':
        mean = cfg.dataset.stats.mean
        std = cfg.dataset.stats.std
        norm_fun = transforms.Normalize(mean=mean, std=std)

    def f(batch):
        batch = default_collate(batch)
        batch['image'] = norm_fun(batch['image'].float())
        return batch
    return f


def uncollate_fn(cfg):
    def to_shape(t1, t2):
        t1 = t1[None].repeat(t2.shape[0], 1)
        t1 = t1.view((t2.shape[:2] + (1, 1)))
        return t1

    def denorm(tensor, mean, std):
        # get stats
        mean = torch.tensor(mean).to(cfg.device)
        std = torch.tensor(std).to(cfg.device)
        # denorm
        return (tensor * to_shape(std, tensor)) + to_shape(mean, tensor)

    def f(sr, lr):
        stats = cfg.dataset.stats
        if sr is not None:
            sr = denorm(sr.float(), stats.mean, stats.std)
        lr = denorm(lr.float(), stats.mean, stats.std)
        return sr, lr

    return f


def printable(cfg):
    def to_shape(t1, t2):
        t1 = t1[None].repeat(t2.shape[0], 1)
        t1 = t1.view(t1.shape + (1, 1))
        return t1

    def _printable(tensor, stats):
        max_ = to_shape(torch.tensor(
            stats.max), tensor).to(cfg.device)
        min_ = to_shape(torch.tensor(
            stats.min), tensor).to(cfg.device)

        # printable
        return (tensor - min_) / (max_ - min_)

    def f(sr, lr):
        stats = cfg.dataset.stats
        #  batch = default_collate(batch)
        if sr is not None:
            sr = _printable(sr, stats)
        lr = _printable(lr, stats)
        return sr, lr

    return f


def get_mean_std(train_dloader, cfg):
    image_ms = MeanStd(device=cfg.device)

    for index, batch in tqdm(
            enumerate(train_dloader), total=len(train_dloader)):
        image = batch["image"].to(device=cfg.device, non_blocking=True).float()

        image_ms(image)

    image_mean, image_std = image_ms.get_mean_std()
    image_min, image_max = image_ms.get_min_max()

    print('Image (mean, std, min, max)')
    print('mean: {},'.format(image_mean))
    print('std: {},'.format(image_std))
    print('min: {},'.format(image_min))
    print('max: {}'.format(image_max))
