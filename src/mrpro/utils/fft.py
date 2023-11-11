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


def kspace_to_image(kdat: torch.Tensor, dim: tuple[int, ...] = (-1, -2, -3)) -> torch.Tensor:
    """IFFT from k-space to image space.

    Parameters
    ----------
    kdat
        k-space data on Cartesian grid
    dim, optional
        dim along which iFFT is applied, by default last three dimensions (-1, -2, -3)

    Returns
    -------
        iFFT of kdat
    """
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(kdat, dim=dim), dim=dim, norm='ortho'), dim=dim)


def image_to_kspace(idat: torch.Tensor, dim: tuple[int, ...] = (-1, -2, -3)) -> torch.Tensor:
    """FFT from image space to k-space.

    Parameters
    ----------
    idat
        image data on Cartesian grid
    dim, optional
        dim along which iFFT is applied, by default last three dimensions (-1, -2, -3)

    Returns
    -------
        FFT of idat
    """
    return torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(idat, dim=dim), dim=dim, norm='ortho'), dim=dim)
