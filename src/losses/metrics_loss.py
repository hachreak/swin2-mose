import piq

from metrics import _cc_single_torch
from utils import load_fun


def norm_0_to_1(fun):
    def wrapper(cfg):
        dset = cfg.dataset
        use_minmax = cfg.dataset.get('stats').get('use_minmax', False)
        denorm = load_fun(dset.get('denorm'))(
                cfg,
                hr_name=cfg.dataset.hr_name,
                lr_name=cfg.dataset.lr_name)
        evaluable = load_fun(dset.get('printable'))(
                cfg,
                hr_name=cfg.dataset.hr_name,
                lr_name=cfg.dataset.lr_name,
                filter_outliers=False,
                use_minmax=use_minmax)

        rfun = fun(cfg)

        def f(sr, hr):
            hr, sr, _ = evaluable(*denorm(hr, sr, None))
            sr = sr.clamp(min=0, max=1)
            hr = hr.clamp(min=0, max=1)
            return rfun(sr, hr)

        return f
    return wrapper


@norm_0_to_1
def ssim_loss(cfg):
    criterion = piq.SSIMLoss()

    def f(sr, hr):
        return criterion(sr, hr)

    return f


def cc_loss(sr, hr):
    cc_value = _cc_single_torch(sr, hr)
    return 1 - ((cc_value + 1) * .5)
