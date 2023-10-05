"""Numerical phantom with ellipses."""

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

from mrpro.phantoms.phantom_elements import EllipsePars


class EllipsePhantom:
    """Numerical phantom as the sum of different ellipses.

    Parameters
    ----------
        ellipses
            ellipses defined by their center, radii and intensity
    """

    def __init__(
        self,
        ellipses: list[EllipsePars] = [
            EllipsePars(center_x=0.2, center_y=0.2, radius_x=0.1, radius_y=0.25, intensity=1),
            EllipsePars(center_x=0.1, center_y=-0.1, radius_x=0.3, radius_y=0.1, intensity=2),
            EllipsePars(center_x=-0.2, center_y=0.2, radius_x=0.18, radius_y=0.25, intensity=4),
        ],
    ):
        self.ellipses: list[EllipsePars] = ellipses

    def kspace(self, ky: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
        """Create 2D analytic kspace data based on given k-space locations.

        For a corresponding image with 256 x 256 voxel, the k-space locations should be defined within [-128, 127]

        The Fourier representation of ellipses can be analytically described by Bessel functions. Further information
        and derivations can be found e.g. here: https://doi.org/10.1002/mrm.21292

        Parameters
        ----------
        ky
            k-space locations in ky
        kx
            k-space locations in kx (frequency encoding direction). Same shape as ky.
        """
        # kx and ky have to be of same shape
        if kx.shape != ky.shape:
            raise ValueError(f'shape mismatch between kx {kx.shape} and ky {ky.shape}')

        kdat = torch.zeros_like(kx, dtype=torch.complex64)
        for el in self.ellipses:
            arg = torch.sqrt((el.radius_x * 2) ** 2 * kx**2 + (el.radius_y * 2) ** 2 * ky**2)
            arg[arg < 1e-6] = 1e-6  # avoid zeros

            cdat = 2 * 2 * el.radius_x * el.radius_y * 0.5 * torch.special.bessel_j1(torch.pi * arg) / arg
            kdat += torch.exp(-1j * 2 * torch.pi * (el.center_x * kx + el.center_y * ky)) * cdat * el.intensity

        # Scale k-space data by factor sqrt(number of points) to ensure correct scaling after FFT with
        # normalization "ortho". See e.g. https://docs.scipy.org/doc/scipy/tutorial/fft.html
        kdat *= np.sqrt(torch.numel(kdat))
        return kdat

    def image_space(self, ny: int, nx: int) -> torch.Tensor:
        """Create image representation of phantom.

        Parameters
        ----------
        ny
            Number of voxel along y direction
        nx
            Number of voxel along x direction
        """
        # Calculate image representation of phantom
        ix, iy = torch.meshgrid(
            torch.linspace(-nx // 2, nx // 2 - 1, nx), torch.linspace(-ny // 2, ny // 2 - 1, ny), indexing='xy'
        )

        idat = torch.zeros((ny, nx), dtype=torch.complex64)
        for el in self.ellipses:
            curr_el = torch.zeros_like(idat)
            curr_el[
                ((ix / nx - el.center_x) ** 2 / el.radius_x**2 + (iy / ny - el.center_y) ** 2 / el.radius_y**2) <= 1
            ] = el.intensity
            idat += curr_el

        return idat
