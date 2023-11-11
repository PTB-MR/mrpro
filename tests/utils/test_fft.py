"""Tests for image space - k-space transformations."""

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
import pytest
import torch

from mrpro.utils.fft import image_to_kspace
from mrpro.utils.fft import kspace_to_image


@pytest.mark.parametrize('npoints, a', [(100, 20), (300, 20)])
def test_kspace_to_image(npoints, a):
    """Test k-space to image transformation using a Gaussian."""
    # Utilize that a Fourier transform of a Gaussian function is given by
    # F(exp(-x^2/a)) = sqrt(pi*a)exp(-a*pi^2k^2)

    # Define k-space between [-1, 1) and image space accordingly
    dk = 2 / npoints
    k = torch.linspace(-1, 1 - dk, npoints)
    dx = 1 / 2
    x = torch.linspace(-1 / (2 * dk), 1 / (2 * dk) - dx, npoints)

    # Create Gaussian function in k-space and image space
    igauss = torch.exp(-(x**2) / a).to(torch.complex64)
    kgauss = np.sqrt(torch.pi * a) * torch.exp(-a * torch.pi**2 * k**2).to(torch.complex64)

    # Transform k-space to image
    kgauss_fft = kspace_to_image(kgauss, dim=(0,))

    # Scaling to "undo" fft scaling
    kgauss_fft *= 2 / np.sqrt(npoints)
    torch.testing.assert_close(kgauss_fft, igauss)


def test_image_to_kspace_as_inverse():
    """Test if image_to_kspace is the inverse of kspace_to_image."""

    # Create random 3D data set
    npoints = [200, 100, 50]
    idat = torch.randn(*npoints, dtype=torch.complex64)

    # Transform to k-space and back along all three dimensions
    idat_transform = image_to_kspace(kspace_to_image(idat))
    torch.testing.assert_close(idat, idat_transform)
