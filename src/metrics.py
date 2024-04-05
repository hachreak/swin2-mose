# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import collections.abc
import math
import typing
import warnings
from itertools import repeat
from typing import Any

import cv2
import numpy as np
import torch
import piq
from numpy import ndarray
from scipy.io import loadmat
from scipy.ndimage.filters import convolve
from scipy.special import gamma
from torch import nn
from torch.nn import functional as F


class piq_metric(object):
    def __init__(self, cfg):
        pass

    def to(self, device):
        return self

    def __call__(self, x, y):
        x = x.clone()
        x[x < 0] = 0.
        x[x > 1] = 1.
        return self._metric(x, y)

    def _metric(self, x, y):
        pass


class piq_psnr(piq_metric):
    def _metric(self, x, y):
        return piq.psnr(x, y)


class piq_ssim(piq_metric):
    def _metric(self, x, y):
        return piq.ssim(x, y)


class piq_rmse(piq_metric):
    def __init__(self, cfg):
        self.mse = nn.MSELoss()

    def _metric(self, x, y):
        return torch.sqrt(self.mse(x, y))


def _ergas_single_torch(raw_tensor: torch.Tensor,
                        dst_tensor: torch.Tensor):
    """
    Compute the ERGAS (Erreur Relative Globale Adimensionnelle De Synthèse) metric for a pair of input tensors.

    ERGAS measures the relative global error of synthesis for remote sensing or image processing tasks.
    It evaluates the quality of an output image concerning a reference image, taking into account spectral bands.

    Args:
        raw_tensor (torch.Tensor): The image tensor to be compared (typically the reconstructed image).
        dst_tensor (torch.Tensor): The reference image tensor.

    Returns:
        ERGAS (torch.Tensor): The ERGAS metric score.

    """
    # Compute the number of spectral bands
    N_spectral = raw_tensor.shape[1]

    # Reshape images for processing
    raw_tensor_reshaped = raw_tensor.view(N_spectral, -1)
    dst_tensor_reshaped = dst_tensor.view(N_spectral, -1)
    N_pixels = raw_tensor_reshaped.shape[1]

    # Assuming HR size is 256x256 and LR size is 128x128
    hr_size = torch.tensor(256).cuda()
    lr_size = torch.tensor(128).cuda()

    # Calculate the beta value
    beta = (hr_size / lr_size).cuda()

    # Calculate RMSE of each band
    rmse = torch.sqrt(torch.nansum((dst_tensor_reshaped - raw_tensor_reshaped) ** 2, dim=1) / N_pixels)
    mu_ref = torch.mean(dst_tensor_reshaped, dim=1)

    # Calculate ERGAS
    ERGAS = 100 * (1 / beta ** 2) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2) / N_spectral)

    return ERGAS


class ERGAS(nn.Module):
    """
    PyTorch implementation of ERGAS (Erreur Relative Globale Adimensionnelle De Synthèse) metric.

    ERGAS measures the relative global error of synthesis for remote sensing or image processing tasks.
    It evaluates the quality of an output image concerning a reference image, taking into account spectral bands.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(raw_tensor, dst_tensor):
            Compute ERGAS metric between two input tensors representing images.

    Example:
        ergas_calculator = ERGAS()
        raw_image = torch.tensor(...)  # Replace with your raw image data
        dst_image = torch.tensor(...)  # Replace with your reference image data
        ergas_score = ergas_calculator(raw_image, dst_image)
        print(f"ERGAS Score: {ergas_score.item()}")

    """

    def __init__(self):
        super().__init__()

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
        """
        Compute ERGAS metric between two input tensors representing images.

        Args:
            raw_tensor (torch.Tensor): The image tensor to be compared (typically the reconstructed image).
            dst_tensor (torch.Tensor): The reference image tensor.

        Returns:
            ergas_metrics (torch.Tensor): The ERGAS metric score.

        Note:
            ERGAS measures the relative global error of synthesis for remote sensing or image processing tasks.
            It evaluates the quality of an output image concerning a reference image, taking into account spectral bands.

        """
        ergas_metrics = _ergas_single_torch(raw_tensor, dst_tensor)
        return ergas_metrics


def _cc_single_torch(raw_tensor: torch.Tensor,
                     dst_tensor: torch.Tensor):

    """
    Compute the Cross-Correlation (CC) metric between two input tensors representing images.

    CC measures the similarity between two images by calculating the cross-correlation coefficient between spectral bands.

    Args:
        raw_tensor (torch.Tensor): The image tensor to be compared.
        dst_tensor (torch.Tensor): The reference image tensor.

    Returns:
        CC (torch.Tensor): The Cross-Correlation (CC) metric score.

    """
    N_spectral = raw_tensor.shape[1]

    # Reshaping fused and reference data
    raw_tensor_reshaped = raw_tensor.view(N_spectral, -1)
    dst_tensor_reshaped = dst_tensor.view(N_spectral, -1)

    # Calculating mean value
    mean_raw = torch.mean(raw_tensor_reshaped, 1).unsqueeze(1)
    mean_dst = torch.mean(dst_tensor_reshaped, 1).unsqueeze(1)

    CC = torch.sum((raw_tensor_reshaped - mean_raw) * (dst_tensor_reshaped - mean_dst), 1) / torch.sqrt(
        torch.sum((raw_tensor_reshaped - mean_raw) ** 2, 1) * torch.sum((dst_tensor_reshaped - mean_dst) ** 2, 1))

    CC = torch.mean(CC)

    return CC


class CC(nn.Module):
    """
    PyTorch implementation of the Cross-Correlation (CC) metric for image similarity.

    CC measures the similarity between two images by calculating the cross-correlation coefficient between spectral bands.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(raw_tensor, dst_tensor):
            Compute the Cross-Correlation (CC) metric between two input tensors representing images.

    Example:
        cc_calculator = CC()
        raw_image = torch.tensor(...)  # Replace with your raw image data
        dst_image = torch.tensor(...)  # Replace with your reference image data
        cc_score = cc_calculator(raw_image, dst_image)
        print(f"Cross-Correlation Score: {cc_score.item()}")

    """

    def __init__(self):
        super().__init__()

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
        """
        Compute the Cross-Correlation (CC) metric between two input tensors representing images.

        Args:
            raw_tensor (torch.Tensor): The image tensor to be compared.
            dst_tensor (torch.Tensor): The reference image tensor.

        Returns:
            cc_metrics (torch.Tensor): The Cross-Correlation (CC) metric score.

        Note:
            CC measures the similarity between two images by calculating the cross-correlation coefficient between spectral bands.

        """
        cc_metrics = _cc_single_torch(raw_tensor, dst_tensor)
        return cc_metrics


def _sam_single_torch(raw_tensor: torch.Tensor,
                      dst_tensor: torch.Tensor):
    """
    Compute the Spectral Angle Mapper (SAM) metric between two input tensors representing images.

    SAM measures the spectral similarity between two images by calculating the spectral angle between corresponding pixels.

    Args:
        raw_tensor (torch.Tensor): The image tensor to be compared.
        dst_tensor (torch.Tensor): The reference image tensor.

    Returns:
        SAM (torch.Tensor): The Spectral Angle Mapper (SAM) metric score.

    """
    # Compute the number of spectral bands
    N_spectral = raw_tensor.shape[1]

    # Reshape fused and reference data
    raw_tensor_reshaped = raw_tensor.view(N_spectral, -1)
    dst_tensor_reshaped = dst_tensor.view(N_spectral, -1)
    N_pixels = raw_tensor_reshaped.shape[1]

    # Calculate the inner product
    inner_prod = torch.nansum(raw_tensor_reshaped * dst_tensor_reshaped, 0)
    raw_norm = torch.nansum(raw_tensor_reshaped ** 2, dim=0).sqrt()
    dst_norm = torch.nansum(dst_tensor_reshaped ** 2, dim=0).sqrt()

    # Calculate SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (raw_norm * dst_norm))) / N_pixels)

    return SAM


class SAM(nn.Module):
    """
    PyTorch implementation of the Spectral Angle Mapper (SAM) metric for spectral similarity.

    SAM measures the spectral similarity between two images by calculating the spectral angle between corresponding pixels.

    Args:
        None

    Attributes:
        None

    Methods:
        forward(raw_tensor, dst_tensor):
            Compute the Spectral Angle Mapper (SAM) metric between two input tensors representing images.

    Example:
        sam_calculator = SAM()
        raw_image = torch.tensor(...)  # Replace with your raw image data
        dst_image = torch.tensor(...)  # Replace with your reference image data
        sam_score = sam_calculator(raw_image, dst_image)
        print(f"Spectral Angle Mapper Score: {sam_score.item()}")

    """

    def __init__(self):
        super().__init__()

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
        """
        Compute the Spectral Angle Mapper (SAM) metric between two input tensors representing images.

        Args:
            raw_tensor (torch.Tensor): The image tensor to be compared.
            dst_tensor (torch.Tensor): The reference image tensor.

        Returns:
            sam_metrics (torch.Tensor): The Spectral Angle Mapper (SAM) metric score.

        Note:
            SAM measures the spectral similarity between two images by calculating the spectral angle between corresponding pixels.

        """
        sam_metrics = _sam_single_torch(raw_tensor, dst_tensor)
        return sam_metrics
