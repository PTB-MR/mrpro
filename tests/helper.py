"""Helper/Utilities for test functions."""

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

import torch


def kspace_to_image(kdat, dim=(-1, -2)):
    """IFFT from k-space to image space.

    Parameters
    ----------
    kdat
        k-space data on Cartesian grid
    dim, optional
        dim along which iFFT is applied, by default last two dimensions (-1, -2)

    Returns
    -------
        FFT of kdat
    """
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(kdat, dim=dim), dim=dim, norm='ortho'), dim=dim)


def rel_image_diff(im1, im2):
    """Calculate mean absolute relative difference between two images.

    Parameters
    ----------
    im1
        first image
    im2
        second image

    Returns
    -------
        mean absolute relative difference between images
    """
    idiff = torch.mean(torch.abs(im1 - im2))
    imean = 0.5 * torch.mean(torch.abs(im1) + torch.abs(im2))
    if imean == 0:
        raise ValueError('average of images should be larger than 0')
    return idiff / imean
