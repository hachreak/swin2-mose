import torch
import importlib
import numpy as np
import random

from enum import Enum
from torchmetrics.classification import MulticlassF1Score


def set_deterministic(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def w_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_fun(fullname):
    path, name = fullname.rsplit('.', 1)
    return getattr(importlib.import_module(path), name)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        self.avg_item = self.avg.tolist()
        fmtstr = "{avg_item" + self.fmt + "}"
        try:
            return fmtstr.format(**self.__dict__)
        except TypeError:
            # print a list of elements
            fmtstr = "{" + self.fmt + "}"
            return ' '.join([
                fmtstr for _ in range(len(self.avg_item))
            ]).format(*self.avg_item)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(
                "Invalid summary type {}".format(self.summary_type))

        return fmtstr.format(**self.__dict__)


class F1AverageMeter(AverageMeter):
    def __init__(self, cfg, average, **kwargs):
        self.cfg = cfg
        self._cfg_average = average
        super().__init__(**kwargs)

    def reset(self):
        super().reset()
        self._to_update = True
        self.fun = MulticlassF1Score(
            len(self.cfg.classes), average=self._cfg_average
        ).to(self.cfg.device)

    @property
    def avg(self):
        if self._to_update:
            self._avg = self.fun.compute()
            self._to_update = False
        return self._avg

    @avg.setter
    def avg(self, value):
        self._avg = value
        self._to_update = False

    def update(self, val, n=1):
        pred, gt = val
        self.fun.update(pred, gt)
        self._to_update = True


def set_required_grad(model, value):
    for parameters in model.parameters():
        parameters.requires_grad = value
