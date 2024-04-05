import torch

from torchvision import transforms
from torch.utils.data import default_collate
from tqdm import tqdm


def collate_fn(cfg, hr_name, lr_name):
    print('collate_fn hr field: ', hr_name)
    print('collate_fn lr field: ', lr_name)

    def do_nothing(x):
        return x
    norm_lr = do_nothing
    norm_hr = do_nothing

    if 'stats' in cfg.dataset and cfg.phase != 'plot_data':
        hr_mean = cfg.dataset.stats.get(hr_name).mean
        hr_std = cfg.dataset.stats.get(hr_name).std
        lr_mean = cfg.dataset.stats.get(lr_name).mean
        lr_std = cfg.dataset.stats.get(lr_name).std
        norm_hr = transforms.Normalize(mean=hr_mean, std=hr_std)
        norm_lr = transforms.Normalize(mean=lr_mean, std=lr_std)

    def f(batch):
        batch = default_collate(batch)

        lr = norm_lr(batch[lr_name].float())
        hr = norm_hr(batch[hr_name].float())

        return {'lr': lr, 'hr': hr}
    return f


def uncollate_fn(cfg, hr_name, lr_name):
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

    def f(hr=None, sr=None, lr=None):
        stats_hr = cfg.dataset.stats.get(hr_name)
        stats_lr = cfg.dataset.stats.get(lr_name)

        if hr is not None:
            hr = denorm(hr, stats_hr.mean, stats_hr.std)
        if sr is not None:
            sr = denorm(sr, stats_hr.mean, stats_hr.std)
        if lr is not None:
            lr = denorm(lr, stats_lr.mean, stats_lr.std)

        return hr, sr, lr

    return f


def printable(cfg, hr_name, lr_name, filter_outliers=False, use_minmax=False):
    def to_shape(t1, t2):
        t1 = t1[None].repeat(t2.shape[0], 1)
        t1 = t1.view(t1.shape + (1, 1))
        return t1

    def _printable(tensor, stats):
        if use_minmax:
            max_ = to_shape(torch.tensor(
                stats.max), tensor).to(cfg.device)
            min_ = to_shape(torch.tensor(
                stats.min), tensor).to(cfg.device)
        else:
            # get stats
            mean = torch.tensor(stats.mean).to(cfg.device)
            std = torch.tensor(stats.std).to(cfg.device)
            # compute min and max
            max_ = to_shape(mean + std, tensor)
            min_ = to_shape(mean - std, tensor)

        # fitler outliers if needed to visualize them
        if filter_outliers:
            tensor = torch.min(torch.max(tensor, min_), max_)
        # printable
        return (tensor - min_) / (max_ - min_)

    def f(hr=None, sr=None, lr=None):
        stats_hr = cfg.dataset.stats.get(hr_name)
        stats_lr = cfg.dataset.stats.get(lr_name)

        if hr is not None:
            hr = _printable(hr, stats_hr)
        if sr is not None:
            sr = _printable(sr, stats_hr)
        if lr is not None:
            lr = _printable(lr, stats_lr)

        return hr, sr, lr

    return f


class MeanStd(object):
    def __init__(self, device):
        self.channels_sum = 0
        self.channels_sqrd_sum = 0
        self.num_batches = 0
        self.max = torch.full((4,), -torch.inf, device=device)
        self.min = torch.full((4,), torch.inf, device=device)

    def __call__(self, data):
        # max
        d_perm = data.permute(1, 0, 2, 3)
        d_max = d_perm.reshape(data.shape[1], -1).max(dim=1)[0]
        self.max = torch.where(d_max > self.max, d_max, self.max)
        # min
        d_min = d_perm.reshape(data.shape[1], -1).min(dim=1)[0]
        self.min = torch.where(d_min <= self.min, d_min, self.min)
        # mean, std
        self.channels_sum += torch.mean(data, dim=[0, 2, 3])
        self.channels_sqrd_sum += torch.mean(data**2, dim=[0, 2, 3])
        self.num_batches += 1

    def get_mean_std(self):
        mean = self.channels_sum / self.num_batches
        std = (self.channels_sqrd_sum / self.num_batches - mean**2) ** 0.5
        return mean.tolist(), std.tolist()

    def get_min_max(self):
        return self.min.tolist(), self.max.tolist()


def get_mean_std(train_dloader, cfg):
    hr_ms = MeanStd(device=cfg.device)
    lr_ms = MeanStd(device=cfg.device)

    for index, batch in tqdm(
            enumerate(train_dloader), total=len(train_dloader)):
        hr = batch["hr"].to(device=cfg.device, non_blocking=True).float()
        lr = batch["lr"].to(device=cfg.device, non_blocking=True).float()

        hr_ms(hr)
        lr_ms(lr)

    hr_mean, hr_std = hr_ms.get_mean_std()
    lr_mean, lr_std = lr_ms.get_mean_std()
    hr_min, hr_max = hr_ms.get_min_max()
    lr_min, lr_max = lr_ms.get_min_max()

    print('HR (mean, std, min, max)')
    print('mean: {},'.format(hr_mean))
    print('std: {},'.format(hr_std))
    print('min: {},'.format(hr_min))
    print('max: {}'.format(hr_max))
    print('LR (mean, std, min, max)')
    print('mean: {},'.format(lr_mean))
    print('std: {},'.format(lr_std))
    print('min: {},'.format(lr_min))
    print('max: {}'.format(lr_max))
