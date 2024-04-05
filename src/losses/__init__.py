from torch import nn

from . import metrics_loss


def build_losses(cfg):
    losses = {}
    if cfg.losses.get('with_ce_criterion', False):
        losses['ce_criterion'] = nn.CrossEntropyLoss()
    if cfg.losses.get('with_pixel_criterion', False):
        losses['pixel_criterion'] = nn.MSELoss()
    if cfg.losses.get('with_cc_criterion', False):
        losses['cc_criterion'] = metrics_loss.cc_loss
    if cfg.losses.get('with_ssim_criterion', False):
        losses['ssim_criterion'] = metrics_loss.ssim_loss(cfg)
    print(losses)
    return losses
