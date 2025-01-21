"""Tests for Fast Fourier Operator class."""

import numpy as np
import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.operators import FastFourierOp

from tests import RandomGenerator, dotproduct_adjointness_test


@pytest.mark.parametrize(('npoints', 'a'), [(100, 20), (300, 20)])
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
    ff_op = FastFourierOp(dim=(0,))
    (igauss_fwd,) = ff_op(igauss)

    # Scaling to "undo" fft scaling
    igauss_fwd *= np.sqrt(npoints) / 2
    torch.testing.assert_close(igauss_fwd, kgauss)


@pytest.mark.parametrize(
    ('encoding_matrix', 'recon_matrix'),
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)
def test_fast_fourier_op_adjoint(encoding_matrix, recon_matrix):
    """Test adjointness of Fast Fourier Op."""

    # Create test data
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(recon_matrix)
    v = generator.complex64_tensor(encoding_matrix)

    # Create operator and apply
    ff_op = FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix)
    dotproduct_adjointness_test(ff_op, u, v)


def test_fast_fourier_op_spatial_dim():
    """Test for equal results if matrices are spatial dimension or lists"""
    # Create test data
    recon_matrix = SpatialDimension(z=101, y=201, x=61)
    encoding_matrix = SpatialDimension(z=14, y=220, x=61)
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(size=(3, 2, recon_matrix.z, recon_matrix.y, recon_matrix.x))
    v = generator.complex64_tensor(size=(3, 2, encoding_matrix.z, encoding_matrix.y, encoding_matrix.x))
    # these should not matter and are set to arbitrary values
    recon_matrix.x = -13
    encoding_matrix.x = -13
    ff_op_list = FastFourierOp(
        recon_matrix=[recon_matrix.y, recon_matrix.z],
        encoding_matrix=[encoding_matrix.y, encoding_matrix.z],
        dim=(-2, -3),
    )
    ff_op_spatialdim = FastFourierOp(
        recon_matrix=recon_matrix,
        encoding_matrix=encoding_matrix,
        dim=(-2, -3),
    )
    assert torch.equal(*ff_op_list(u), *ff_op_spatialdim(u))
    assert torch.equal(*ff_op_list.H(v), *ff_op_spatialdim.H(v))


def test_fast_fourier_op_onematrix():
    recon_matrix = SpatialDimension(z=101, y=201, x=61)
    encoding_matrix = SpatialDimension(z=14, y=220, x=61)
    with pytest.raises(ValueError, match='None'):
        FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=None)
    with pytest.raises(ValueError, match='None'):
        FastFourierOp(recon_matrix=None, encoding_matrix=encoding_matrix)


def test_invalid_dim():
    """Tests that dims are in (-3,-2,-1) if recon_matrix
    or encoding_matrix is SpatialDimension"""

    recon_matrix = SpatialDimension(z=101, y=201, x=61)
    encoding_matrix = SpatialDimension(z=14, y=220, x=61)

    with pytest.raises(NotImplementedError, match='recon_matrix'):
        FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=None, dim=(-4, -2, -1))

    with pytest.raises(NotImplementedError, match='encoding_matrix'):
        FastFourierOp(recon_matrix=None, encoding_matrix=encoding_matrix, dim=(-4, -2, -1))
