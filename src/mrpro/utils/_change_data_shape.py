"""Wrapper for FFT and IFFT."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy as np
import torch
import torch.nn.functional as F


def change_data_shape(dat: torch.Tensor, dat_shape_new: tuple[int, ...]) -> torch.Tensor:
    """Change shape of data by cropping or zero-padding.

    Parameters
    ----------
    dat
        data
    dat_shape_new
        desired shape of data

    Returns
    -------
        data with shape dat_shape_new
    """
    s = list(dat.shape)
    # Padding
    npad = [0] * (2 * len(s))

    for idx in range(len(s)):
        if s[idx] != dat_shape_new[idx]:
            dim_diff = dat_shape_new[idx] - s[idx]
            # This is needed to ensure that padding and cropping leads to the same asymetry for odd shape differences
            npad[2 * idx] = np.sign(dim_diff) * (np.abs(dim_diff) // 2)
            npad[2 * idx + 1] = dat_shape_new[idx] - (s[idx] + npad[2 * idx])

    # Pad (positive npad) or crop (negative npad)
    # npad has to be reversed because pad expects it in reversed order
    if not torch.all(torch.tensor(npad) == 0):
        dat = F.pad(dat, npad[::-1])
    return dat
