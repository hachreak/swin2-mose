import torch

from torchgeo.models import FarSeg
from easydict import EasyDict
from torch import nn
from torch.nn import functional as F

from chk_loader import load_state_dict_model_only
from config import load_config
from super_res.model import build_model as super_res_build_model
from utils import set_required_grad


class UpScaler(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        u_cfg = cfg.semantic_segm.upscaler

        # load dl upscaler
        if u_cfg.get('chk') is not None:
            print('load super_res model config file ', u_cfg.config)
            model_cfg = EasyDict(load_config(u_cfg.config))
            model_cfg.device = cfg.device
            # load super_res
            print("loading super_res model")
            self.super_res = super_res_build_model(model_cfg)
            # load checkpoint
            print("loading super_res chk file {}".format(u_cfg.chk))
            swin_checkpoint = torch.load(u_cfg.chk)
            load_state_dict_model_only(self.super_res, swin_checkpoint)
            # put eval mode
            print('set super_res eval mode')
            set_required_grad(self.super_res, False)

    def forward(self, x):
        with torch.no_grad():
            x = self.super_res.forward_backbone(x)
        return x


class SemanticSegm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ssegm = cfg.semantic_segm

        # padding before and after
        self.pad_before = ssegm.pad_before
        self.pad_after = self.pad_before
        if 'pad_after' in ssegm:
            self.pad_after = ssegm.pad_after

        # segmentation model
        if ssegm.type == 'FarSeg':
            self.model = FarSeg(classes=len(cfg.classes), **ssegm.model)
            out_ch = self.model.backbone.conv1.out_channels
            self.model.backbone.conv1 = nn.Conv2d(
                ssegm.in_channels, out_ch, kernel_size=7, stride=2, padding=3
            )

        # super res model
        if 'upscaler' in ssegm:
            self.upscaler = UpScaler(cfg)
        # or conv [4, H, W] -> [90, H, W] for model++
        if 'conv_up' in ssegm:
            print('model++ with conv_up')
            print(ssegm.conv_up)
            kernel_size = ssegm.conv_up.get('kernel_size', 1)
            padding = ssegm.conv_up.get('padding', 0)
            self.conv_up = nn.Conv2d(
                ssegm.conv_up.in_ch, ssegm.conv_up.middle_ch,
                kernel_size=kernel_size, padding=padding)
            self.model.backbone.conv1 = nn.Conv2d(
                ssegm.conv_up.middle_ch, ssegm.conv_up.out_ch,
                kernel_size=7, stride=2, padding=3)

    def forward(self, x):
        # upscale input if super res model is defined
        if hasattr(self, 'upscaler'):
            x = self.upscaler(x)
            if not torch.is_tensor(x):
                # remove loss_moe value
                x, _ = x

        pad_x = F.pad(x, self.pad_before, "constant", 0)

        # or use conv to increase channels (for baseline++)
        if hasattr(self, 'conv_up'):
            pad_x = self.conv_up(pad_x)

        # semantic segmantion model
        pad_x = self.model(pad_x)
        x = pad_x[...,
                  self.pad_after[0]:-self.pad_after[1],
                  self.pad_after[2]:-self.pad_after[3]]

        return x


def build_model(cfg):
    return SemanticSegm(cfg).to(cfg.device)
