import os
import torch
import matplotlib.pyplot as plt
import imgproc

from tqdm import tqdm

from validation import build_eval_metrics, load_eval_method
from utils import load_fun


def plot_images(img, out_path, basename, fname, dpi):
    # dir: out_dir / basename
    out_path = os.path.join(out_path, basename)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    img = img.squeeze(0)
    for i in range(img.shape[0]):
        plt.imshow(imgproc.tensor_to_image(img[i].detach(), False, False))
        plt.axis('off')
        out_fname = os.path.join(out_path, '{}_{}.png'.format(fname, i))
        plt.savefig(out_fname, dpi=dpi)
        plt.close()


def main(cfg):
    # Load model state dict
    model = load_eval_method(cfg)

    # manipulation function for data
    use_minmax = cfg.dataset.get('stats').get('use_minmax', False)
    denorm = load_fun(cfg.dataset.get('denorm'))(
            cfg,
            hr_name=cfg.dataset.hr_name,
            lr_name=cfg.dataset.lr_name)
    evaluable = load_fun(cfg.dataset.get('printable'))(
            cfg,
            hr_name=cfg.dataset.hr_name,
            lr_name=cfg.dataset.lr_name,
            filter_outliers=False,
            use_minmax=use_minmax)
    printable = load_fun(cfg.dataset.get('printable'))(
            cfg,
            hr_name=cfg.dataset.hr_name,
            lr_name=cfg.dataset.lr_name,
            filter_outliers=False,
            use_minmax=use_minmax)

    # Create a folder of super-resolution experiment results
    out_path = os.path.join(cfg.output, 'images_{}'.format(cfg.epoch))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # move to device
    model.to(cfg.device)
    model.eval()
    # Load dataset
    load_dataset_fun = load_fun(cfg.dataset.get(
        'load_dataset', 'datasets.sen2venus.load_dataset'))
    _, val_dloader, _ = load_dataset_fun(cfg, only_test=True)
    # Define metrics for evaluate each image
    metrics = build_eval_metrics(cfg)

    indices = dict.fromkeys(metrics.keys(), None)
    iterations = min(cfg.num_images, len(val_dloader))

    with torch.no_grad():
        for index, batch in tqdm(
                enumerate(val_dloader), total=iterations,
                desc='%d Images' % (iterations)):

            if index >= iterations:
                break

            hr = batch["hr"].to(device=cfg.device, non_blocking=True)
            lr = batch["lr"].to(device=cfg.device, non_blocking=True)

            sr = model(lr)

            if not torch.is_tensor(sr):
                sr, _ = sr

            # denormalize to original values
            hr_dn, sr_dn, lr_dn = denorm(hr, sr, lr)

            # normalize [0, 1] using also the outliers to evaluate
            hr_eval, sr_eval, lr_eval = evaluable(hr_dn, sr_dn, lr_dn)
            # compute metrics
            for k, fun in metrics.items():
                for i in range(len(sr)):
                    res = fun(sr_eval[i][None], hr_eval[i][None]).detach()
                    indices[k] = res if not res.shape else res.squeeze(0)

            # normalize [0, 1] removing outliers to have a printable version
            hr, sr, lr = printable(hr_dn, sr_dn, lr_dn)
            # plot images
            plot_images(lr, out_path, 'lr', index, cfg.dpi)
            plot_images(hr, out_path, 'hr', index, cfg.dpi)
            plot_images(sr, out_path, 'sr', index, cfg.dpi)
            plot_images((hr - sr).abs(), out_path, 'delta', index, cfg.dpi)
