import torch
import os
import matplotlib.pyplot as plt
import imgproc

from tqdm import tqdm

from chk_loader import load_checkpoint
from datasets.seasonet import load_dataset
from utils import load_fun


def plot_images(img, out_path, basename, fname, batch_number, dpi):
    # dir: out_dir / basename
    out_path = os.path.join(out_path, basename)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    n_cols = 1
    n_bands = img.shape[1]
    fig, axs = plt.subplots(
        n_bands, n_cols, sharex=True, sharey=True, figsize=(n_bands, 1))
    if img is not None:
        img = img.squeeze(0)
        for i in range(img.shape[0]):
            axs[i].imshow(
                imgproc.tensor_to_image(img[i].detach(), False, False))
            axs[i].axis('off')

    out_fname = os.path.join(out_path, '{}_{}.png'.format(fname, batch_number))
    plt.savefig(out_fname, dpi=dpi)
    plt.close()


def save_fig(output, filename, dpi):
    plt.imshow(output.cpu().numpy().squeeze(0))
    plt.savefig(filename, dpi=dpi, pad_inches=0.01)
    print('image saved: {}'.format(filename))


def plot_classes(image, out_path, basename, index, cfg):
    # dir: out_dir / basename
    out_path = os.path.join(out_path, basename)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for idx, name in enumerate(cfg.classes):
        output = (image == idx).int() * 255.
        unq = output.unique()
        #  import ipdb; ipdb.set_trace()
        if len(unq) > 1 or unq.item() != 0:
            print('class ', idx)
            #  import ipdb; ipdb.set_trace()
            fname = 'image_{:02d}_{:02d}.jpg'.format(index, idx)
            full_name = os.path.join(out_path, fname)
            save_fig(output, full_name, cfg.dpi)


def main(cfg):
    cfg['batch_size'] = 1
    vis = cfg.get('visualize', {})
    # Load dataset
    _, val_dloader, _ = load_dataset(cfg, only_test=True)
    # Initialize model
    model = load_fun(vis.get('model'))(cfg)
    denorm = load_fun(cfg.dataset.get('denorm'))(cfg)
    #  evaluable = load_fun(cfg.dataset.get('printable'))(cfg)
    printable = load_fun(cfg.dataset.get('printable'))(cfg)

    # Load model state dict
    try:
        checkpoint = load_checkpoint(cfg)
        _, _ = load_fun(vis.get('checkpoint'))(model, checkpoint)
    except Exception:
        print('no model checkpoint found')
        exit(0)

    # Create a folder of super-resolution experiment results
    out_path = os.path.join(cfg.output, 'images_{}'.format(cfg.epoch))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model.to(cfg.device)
    model.eval()

    iterations = min(cfg.num_images, len(val_dloader))

    for index, batch in tqdm(
            enumerate(val_dloader), total=iterations,
            desc='%d Images' % (iterations)):

        if index >= iterations:
            break

        batch['image'] = batch['image'].to(cfg.device)
        batch['mask'] = batch['mask'].to(cfg.device)

        out = model(batch['image'])
        preds = out.max(-1)[1]

        plot_classes(preds, out_path, 'preds', index, cfg)
        plot_classes(batch['mask'], out_path, 'gt', index, cfg)

        sr = None
        if not cfg.hide_sr:
            if hasattr(model, 'upscaler'):
                sr = model.upscaler(batch['image'])

                if not torch.is_tensor(sr):
                    sr, _ = sr

        # denormalize to original values
        sr, lr = denorm(sr, batch['image'])

        # normalize [0, 1] removing outliers to have a printable version
        sr, lr = printable(sr, lr)

        # plot images
        plot_images(lr, out_path, 'images', 'lr', index, cfg.dpi)

        if not cfg.hide_sr and sr is not None:
            plot_images(sr, out_path, 'images', 'sr', index, cfg.dpi)
