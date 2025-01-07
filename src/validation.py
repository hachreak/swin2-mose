import os
import torch

from torch import nn
from tqdm import tqdm
from torch.nn import Upsample
from collections import OrderedDict

from utils import AverageMeter, load_fun
from metrics import CC, SAM, ERGAS, piq_psnr, piq_ssim, \
    piq_rmse
from chk_loader import load_checkpoint


def validate(g_model, val_dloader, metrics, epoch, writer, mode, cfg):
    # Put the adversarial network model in validation mode
    g_model.eval()

    avg_metrics = build_avg_metrics(cfg)

    use_minmax = cfg.dataset.get('stats', {}).get('use_minmax', False)
    dset = cfg.dataset
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

    with torch.no_grad():
        for j, batch in tqdm(
                enumerate(val_dloader), total=len(val_dloader),
                desc='Val Epoch: %d / %d' % (epoch + 1, cfg.epochs),
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                       "[{elapsed}<{remaining}, {rate_noinv_fmt}]"):
            hr = batch["hr"].to(device=cfg.device, non_blocking=True)
            lr = batch["lr"].to(device=cfg.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            sr = g_model(lr)

            if not torch.is_tensor(sr):
                sr, _ = sr

            sr = sr.contiguous()

            # denormalize to original values
            hr, sr, lr = denorm(hr, sr, lr)

            # normalize [0, 1] using also the outliers to evaluate
            hr, sr, lr = evaluable(hr, sr, lr)

            # Statistical loss value for terminal data output
            for k, fun in metrics.items():
                for i in range(len(sr)):
                    avg_metrics[k].update(fun(sr[i][None], hr[i][None]))

    if writer is not None:
        for k, v in avg_metrics.items():
            writer.add_scalar("{}/{}".format(mode, k), v.avg.item(), epoch+1)

    if cfg.get('eval_return_to_train', True):
        g_model.train()
    return avg_metrics


def build_eval_metrics(cfg):
    # Create an IQA evaluation model
    metrics = {
        'psnr_model': piq_psnr(cfg),
        'ssim_model': piq_ssim(cfg),
        'cc_model': CC(),
        'rmse_model': piq_rmse(cfg),
        'sam_model': SAM(),
        'ergas_model': ERGAS(),
    }

    for k in metrics.keys():
        metrics[k] = metrics[k].to(cfg.device)

    return metrics


def build_avg_metrics(cfg):
    save_values = cfg.metrics_values
    return OrderedDict([
        ('psnr_model', AverageMeter("PIQ_PSNR", ":4.4f",
                                    save_values=save_values)),
        ('ssim_model', AverageMeter("PIQ_SSIM", ":4.4f",
                                    save_values=save_values)),
        ('cc_model', AverageMeter("CC", ":4.4f",
                                  save_values=save_values)),
        ('rmse_model', AverageMeter("PIQ_RMSE", ":4.4f",
                                    save_values=save_values)),
        ('sam_model', AverageMeter("SAM", ":4.4f",
                                   save_values=save_values)),
        ('ergas_model', AverageMeter("ERGAS", ":4.4f",
                                     save_values=save_values)),
    ])


def main(val_dloader, cfg, save_metrics=True):
    model = load_eval_method(cfg)
    print('build eval metrics')
    metrics = build_eval_metrics(cfg)
    result = validate(
        model, val_dloader, metrics, cfg.epoch, None, 'test', cfg)
    if save_metrics:
        do_save_metrics(result, cfg)
    return result


def get_result_filename(cfg):
    output_dir = os.path.join(cfg.output, 'eval')
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, 'results-{:02d}.pt'.format(cfg.epoch))


def do_save_metrics(metrics, cfg):
    filename = get_result_filename(cfg)
    print('save results {}'.format(filename))
    content = {
        'epoch': cfg.epoch,
        'metrics': OrderedDict([
            (k, v.avg) for k, v in metrics.items()
        ])
    }
    if hasattr(metrics['psnr_model'], 'vals'):
        content['values'] = OrderedDict([
            (k, metrics['psnr_model'].values)
            for k, v in metrics.items()
        ])
    torch.save(content, filename)


def load_metrics(cfg):
    filename = get_result_filename(cfg)
    print('load results {}'.format(filename))
    result = torch.load(filename)
    # check if epoch corresponds
    assert result['epoch'] == cfg.epoch
    # build AVG objects
    avg_metrics = build_avg_metrics(cfg)
    for k, v in result['metrics'].items():
        avg_metrics[k].avg = v
    if 'values' in result:
        for k, v in result['values'].items():
            avg_metrics[k].values = v
    return avg_metrics


def print_metrics(metrics):
    names = []
    values = []
    for i, v in enumerate(metrics.values()):
        try:
            names.append(v.name)
            values.append(v)
        except AttributeError:
            # skip for retrocompatibility
            pass
    print(*names)
    print(*values)


def load_eval_method(cfg):
    if cfg.eval_method is None:
        vis = cfg.visualize
        model = load_fun(vis.get('model'))(cfg)
        # Load model state dict
        try:
            checkpoint = load_checkpoint(cfg)
            _, _ = load_fun(vis.get('checkpoint'))(model, checkpoint)
        except Exception as e:
            print(e)
            exit(0)

        return model

    print('load non-dl upsampler: {}'.format(cfg.eval_method))
    return NonDLEvalMethod(cfg)


class NonDLEvalMethod(object):
    def __init__(self, cfg):
        self.upscale_factor = cfg.metrics.upscale_factor
        self.upsampler = Upsample(
            scale_factor=self.upscale_factor,
            mode=cfg.eval_method)

    def __call__(self, x):
        return self.upsampler(x)

    def eval(self):
        pass

    def train(self):
        pass

    def to(self, device):
        return self
