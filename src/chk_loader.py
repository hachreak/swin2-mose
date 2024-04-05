import os
import torch
import numpy as np


def get_last_epoch(filenames):
    epochs = [int(name.split('-')[1].split('.')[0]) for name in filenames]
    return filenames[np.array(epochs).argsort()[-1]]


def load_checkpoint(cfg):
    dir_chk = os.path.join(cfg.output, 'checkpoints')
    if cfg.epoch != -1:
        path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(cfg.epoch))
    else:
        try:
            fnames = os.listdir(dir_chk)
            path = get_last_epoch(fnames)
            path = os.path.join(dir_chk, path)
        except IndexError:
            raise FileNotFoundError()
    # load checkpoint
    print('load file {}'.format(path))
    if not os.path.exists(path):
        raise FileNotFoundError()
    return torch.load(path, map_location=cfg.device)


def load_state_dict_model(model, optimizer, checkpoint):
    print('load model state')
    model.load_state_dict(checkpoint['model_state_dict'])
    print('load optimizer state')
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epoch'] + 1, checkpoint['index']


def load_state_dict_model_only(model, checkpoint):
    print('load model state')
    model.load_state_dict(checkpoint['model_state_dict'])

    return checkpoint['epoch'] + 1, checkpoint['index']


def save_state_dict_model(model, optimizer, epoch, index, cfg):
    # save checkpoint
    n_epoch = epoch + 1
    if (n_epoch) % cfg.snapshot_interval == 0:
        dir_chk = os.path.join(cfg.output, 'checkpoints')
        os.makedirs(dir_chk, exist_ok=True)
        path = os.path.join(dir_chk, 'model-{:02d}.pt'.format(n_epoch))

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'index': index,
        }

        torch.save(checkpoint, path)
