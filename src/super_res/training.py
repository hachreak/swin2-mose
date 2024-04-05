import datetime
import torch

import debug

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from validation import validate, do_save_metrics as save_metrics
from chk_loader import load_checkpoint, load_state_dict_model, \
        save_state_dict_model
from validation import build_eval_metrics
from losses import build_losses
from optim import build_optimizer
from .model import build_model


def train(train_dloader, val_dloader, cfg):

    # Tensorboard
    writer = SummaryWriter(cfg.output + '/tensorboard/train_{}'.format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    # eval every x
    eval_every = cfg.metrics.get('eval_every', 1)

    model = build_model(cfg)
    losses = build_losses(cfg)
    optimizer = build_optimizer(model, cfg)

    begin_epoch = 0
    index = 0
    try:
        checkpoint = load_checkpoint(cfg)
        begin_epoch, index = load_state_dict_model(
            model, optimizer, checkpoint)
    except FileNotFoundError:
        print('no checkpoint found')

    print('build eval metrics')
    metrics = build_eval_metrics(cfg)

    for e in range(begin_epoch, cfg.epochs):
        index = train_epoch(
            model,
            train_dloader,
            losses,
            optimizer,
            e,
            writer,
            index,
            cfg)

        if (e+1) % eval_every == 0:
            result = validate(
                model, val_dloader, metrics, e, writer, 'test', cfg)
            # save result of eval
            cfg.epoch = e+1
            save_metrics(result, cfg)

        save_state_dict_model(model, optimizer, e, index, cfg)


def train_epoch(model, train_dloader, losses, optimizer, epoch, writer,
                index, cfg):
    weights = cfg.losses.weights
    for index, batch in tqdm(
            enumerate(train_dloader, index), total=len(train_dloader),
            desc='Epoch: %d / %d' % (epoch + 1, cfg.epochs)):

        # Transfer in-memory data to CUDA devices to speed up training
        hr = batch["hr"].to(device=cfg.device, non_blocking=True)
        lr = batch["lr"].to(device=cfg.device, non_blocking=True)

        sr = model(lr)

        loss_tracker = {}

        loss_moe = None
        if not torch.is_tensor(sr):
            sr, loss_moe = sr
            if torch.is_tensor(loss_moe):
                loss_tracker['loss_moe'] = loss_moe * weights.moe

        sr = sr.contiguous()

        if 'pixel_criterion' in losses:
            loss_tracker['pixel_loss'] = weights.pixel * \
                losses['pixel_criterion'](sr, hr)

        # cc loss
        if 'cc_criterion' in losses:
            loss_tracker['cc_loss'] = weights.cc * \
                losses['cc_criterion'](sr, hr)

        # ssim loss
        if 'ssim_criterion' in losses:
            loss_tracker['ssim_loss'] = weights.ssim * \
                losses['ssim_criterion'](sr, hr)

        # train
        loss_tracker['train_loss'] = sum(loss_tracker.values())
        optimizer.zero_grad()
        loss_tracker['train_loss'].backward()
        optimizer.step()

        debug.log_hr_stats(lr, sr, hr, writer, index, cfg)
        debug.log_losses(loss_tracker, 'train', writer, index)

    return index
