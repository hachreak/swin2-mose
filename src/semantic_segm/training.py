import datetime

import debug

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from .validation import validate, save_metrics
from chk_loader import load_checkpoint, load_state_dict_model, \
        save_state_dict_model
from optim import build_optimizer
from .model import build_model
from losses import build_losses


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

    for e in range(begin_epoch, cfg.epochs):
        index = train_epoch(
            model, train_dloader, losses, optimizer, e, writer, index, cfg)

        if (e+1) % eval_every == 0:
            result = validate(
                model, val_dloader, e, writer, 'test', cfg)
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

        batch['image'] = batch['image'].to(cfg.device)
        batch['mask'] = batch['mask'].to(cfg.device)

        out = model(batch['image'])

        loss_tracker = {}

        if 'ce_criterion' in losses:
            loss_tracker['ce_loss'] = losses['ce_criterion'](
                out, batch['mask']) * weights.ce

        # train
        loss_tracker['train_loss'] = sum(loss_tracker.values())
        optimizer.zero_grad()
        loss_tracker['train_loss'].backward()
        optimizer.step()

        #  log_stats(writer, index, cfg)
        debug.log_losses(loss_tracker, 'train', writer, index)

    return index
