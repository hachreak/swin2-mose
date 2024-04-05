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
import math
import os
import random
from typing import Any

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
import tifffile


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool):
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    # image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    # i dati non vengono più convertiti in uint8 ma restano in float32
    # se l'immagine è RGB/RGBA fa la permutazione, altrimenti no
    # facendo la permutazione ad un'immagine a 12 canali, il risultato SR sarebbe di dimensione 12x636 e 628 canali
    if 1 < tensor.size()[1] <= 4:
        # image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("float32")
        image = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy().astype("float32")
    else:
        # image = tensor.squeeze(0).mul(255).clamp(0, 255).cpu().numpy().astype("float32")
        image = tensor.squeeze(0).clamp(0, 1).cpu().numpy().astype("float32")

    return image
