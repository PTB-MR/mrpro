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

from collections.abc import Sequence

import numpy as np
import torch
from einops import repeat

from mrpro.data import SpatialDimension
from mrpro.phantoms.phantom_elements import EllipseParameters


class EllipsePhantom:
    """Numerical phantom as the sum of different ellipses.

    Parameters
    ----------
        ellipses
            ellipses defined by their center, radii and intensity.
            if None, defaults to three ellipses
    """

    def __init__(self, ellipses: Sequence[EllipseParameters] | None = None):
        if ellipses is None:
            self.ellipses = [
                EllipseParameters(center_x=0.2, center_y=0.2, radius_x=0.1, radius_y=0.25, intensity=1),
                EllipseParameters(center_x=0.1, center_y=-0.1, radius_x=0.3, radius_y=0.1, intensity=2),
                EllipseParameters(center_x=-0.2, center_y=0.2, radius_x=0.18, radius_y=0.25, intensity=4),
            ]
        else:
            self.ellipses = list(ellipses)

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

        kdata = torch.zeros_like(kx, dtype=torch.complex64)
        for ellipse in self.ellipses:
            arg = torch.sqrt((ellipse.radius_x * 2) ** 2 * kx**2 + (ellipse.radius_y * 2) ** 2 * ky**2)
            arg[arg < 1e-6] = 1e-6  # avoid zeros

            cdata = 2 * 2 * ellipse.radius_x * ellipse.radius_y * 0.5 * torch.special.bessel_j1(torch.pi * arg) / arg
            kdata += (
                torch.exp(-1j * 2 * torch.pi * (ellipse.center_x * kx + ellipse.center_y * ky))
                * cdata
                * ellipse.intensity
            )

        # Scale k-space data by factor sqrt(number of points) to ensure correct scaling after FFT with
        # normalization "ortho". See e.g. https://docs.scipy.org/doc/scipy/tutorial/fft.html
        kdata *= np.sqrt(torch.numel(kdata))
        return kdata

    def image_space(self, image_dimensions: SpatialDimension[int]) -> torch.Tensor:
        """Create image representation of phantom.

        Parameters
        ----------
        image_dimensions
            number of voxels in the image
            This is a 2D simulation so the output will be (1 1 1 image_dimensions.y image_dimensions.x)
        """
        # Calculate image representation of phantom
        ny, nx = image_dimensions.y, image_dimensions.x
        ix, iy = torch.meshgrid(
            torch.linspace(-nx // 2, nx // 2 - 1, nx),
            torch.linspace(-ny // 2, ny // 2 - 1, ny),
            indexing='xy',
        )

        idata = torch.zeros((ny, nx), dtype=torch.complex64)
        for ellipse in self.ellipses:
            in_ellipse = (
                (ix / nx - ellipse.center_x) ** 2 / ellipse.radius_x**2
                + (iy / ny - ellipse.center_y) ** 2 / ellipse.radius_y**2
            ) <= 1
            idata += ellipse.intensity * in_ellipse

        return repeat(idata, 'y x->other coils z y x', other=1, coils=1, z=1)
