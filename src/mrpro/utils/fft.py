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

import torch
import torch.nn.functional as F


def change_image_shape(idat: torch.Tensor, idat_shape_new: tuple[int, ...]) -> torch.Tensor:
    """Change shape of image by cropping or zero-padding.

    Parameters
    ----------
    idat
        image data
    idat_shape_new
        desired shape of image

    Returns
    -------
        image with shape idat_shape_new
    """
    s = list(idat.shape)
    # Padding
    npad = [0] * (2 * len(s))

    for idx in range(len(s)):
        if s[idx] != idat_shape_new[idx]:
            npad[2 * idx] = (idat_shape_new[idx] - s[idx]) // 2
            npad[2 * idx + 1] = idat_shape_new[idx] - (s[idx] + npad[2 * idx])

    # Pad (positive npad) or crop  (negative npad)
    # Npad has to be reversed because pad expects it in reversed order
    if not torch.all(torch.tensor(npad) != 0):
        idat = F.pad(idat, npad[::-1])
    return idat


def kspace_to_image(
    kdat: torch.Tensor, recon_shape: tuple[int, ...] | None = None, dim: tuple[int, ...] = (-1, -2, -3)
) -> torch.Tensor:
    """IFFT from k-space to image space.

    Parameters
    ----------
    kdat
        k-space data on Cartesian grid
    recon_shape, optional
        shape of reconstructed image
    dim, optional
        dim along which iFFT is applied, by default last three dimensions (-1, -2, -3)

    Returns
    -------
        iFFT of kdat
    """
    # FFT
    idat = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(kdat, dim=dim), dim=dim, norm='ortho'), dim=dim)

    # Adapt image size
    if recon_shape is not None:
        s = list(idat.shape)
        for idx, idim in enumerate(dim):
            s[idim] = recon_shape[idx]
        idat = change_image_shape(idat, tuple(s))

    return idat


def image_to_kspace(
    idat: torch.Tensor, encoding_shape: tuple[int, ...] | None = None, dim: tuple[int, ...] = (-1, -2, -3)
) -> torch.Tensor:
    """FFT from image space to k-space.

    Parameters
    ----------
    idat
        image data on Cartesian grid
    encoding_shape, optional
        shape of encoded image
    dim, optional
        dim along which FFT is applied, by default last three dimensions (-1, -2, -3)

    Returns
    -------
        FFT of idat
    """
    # Adapt image size
    if encoding_shape is not None:
        s = list(idat.shape)
        for idx, idim in enumerate(dim):
            s[idim] = encoding_shape[idx]
        idat = change_image_shape(idat, tuple(s))

    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(idat, dim=dim), dim=dim, norm='ortho'), dim=dim)
