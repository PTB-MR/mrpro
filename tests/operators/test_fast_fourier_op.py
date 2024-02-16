"""Tests for Fast Fourier Operator class."""

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

from mrpro.operators import FastFourierOp
from tests import RandomGenerator


@pytest.mark.parametrize('npoints, a', [(100, 20), (300, 20)])
def test_fast_fourier_op_forward(npoints, a):
    """Test Fast Fourier Op transformation using a Gaussian."""
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

    # Transform image to k-space
    FFOp = FastFourierOp(dim=(0,))
    (igauss_fwd,) = FFOp.forward(igauss)

    # Scaling to "undo" fft scaling
    igauss_fwd *= np.sqrt(npoints) / 2
    torch.testing.assert_close(igauss_fwd, kgauss)


@pytest.mark.parametrize(
    'encoding_shape, recon_shape',
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)
def test_fast_fourier_op_adjoint(encoding_shape, recon_shape):
    """Test adjointness of Fast Fourier Op."""

    # Create test data
    generator = RandomGenerator(seed=0)
    x = generator.complex64_tensor(recon_shape)
    y = generator.complex64_tensor(encoding_shape)

    # Create operator and apply
    FFOp = FastFourierOp(recon_shape=recon_shape, encoding_shape=encoding_shape)
    (Ax,) = FFOp.forward(x)
    (AHy,) = FFOp.adjoint(y)

    assert torch.isclose(torch.vdot(Ax.flatten(), y.flatten()), torch.vdot(x.flatten(), AHy.flatten()), rtol=1e-3)
