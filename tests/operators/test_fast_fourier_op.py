"""Tests for Fast Fourier Operator class."""

from collections.abc import Sequence

import numpy as np
import pytest
import torch
from mrpro.data import SpatialDimension
from mrpro.operators import FastFourierOp

from tests import (
    RandomGenerator,
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_fast_fourier_op_and_range_domain(
    recon_matrix: Sequence[int], encoding_matrix: Sequence[int]
) -> tuple[FastFourierOp, torch.Tensor, torch.Tensor]:
    """Create a fast Fourier operator and an element from domain and range."""
    # Create test data
    generator = RandomGenerator(seed=0)
    u = generator.complex64_tensor(recon_matrix)
    v = generator.complex64_tensor(encoding_matrix)

    # Create operator and apply
    ff_op = FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix)
    return ff_op, u, v


@pytest.mark.parametrize(('npoints', 'a'), [(100, 20), (300, 20)])
def test_fast_fourier_op_forward(npoints: int, a: int) -> None:
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


MATRIX_PARAMETERS = pytest.mark.parametrize(
    ('encoding_matrix', 'recon_matrix'),
    [
        ((101, 201, 50), (13, 221, 64)),
        ((100, 200, 50), (14, 220, 64)),
        ((101, 201, 50), (14, 220, 64)),
        ((100, 200, 50), (13, 221, 64)),
    ],
)


@MATRIX_PARAMETERS
def test_fast_fourier_op_adjoint(encoding_matrix: Sequence[int], recon_matrix: Sequence[int]) -> None:
    """Test adjointness of Fast Fourier Op."""
    dotproduct_adjointness_test(*create_fast_fourier_op_and_range_domain(recon_matrix, encoding_matrix))


@MATRIX_PARAMETERS
def test_density_compensation_op_grad(encoding_matrix: Sequence[int], recon_matrix: Sequence[int]) -> None:
    """Test the gradient of the fast Fourier operator."""
    gradient_of_linear_operator_test(*create_fast_fourier_op_and_range_domain(recon_matrix, encoding_matrix))


@MATRIX_PARAMETERS
def test_density_compensation_op_forward_mode_autodiff(
    encoding_matrix: Sequence[int], recon_matrix: Sequence[int]
) -> None:
    """Test forward-mode autodiff of the fast Fourier operator."""
    forward_mode_autodiff_of_linear_operator_test(
        *create_fast_fourier_op_and_range_domain(recon_matrix, encoding_matrix)
    )


def test_fast_fourier_op_spatial_dim() -> None:
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


def test_fast_fourier_op_onematrix() -> None:
    recon_matrix = SpatialDimension(z=101, y=201, x=61)
    encoding_matrix = SpatialDimension(z=14, y=220, x=61)
    with pytest.raises(ValueError, match='None'):
        FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=None)
    with pytest.raises(ValueError, match='None'):
        FastFourierOp(recon_matrix=None, encoding_matrix=encoding_matrix)


def test_invalid_dim() -> None:
    """Tests that dims are in (-3,-2,-1) if recon_matrix
    or encoding_matrix is SpatialDimension"""

    recon_matrix = SpatialDimension(z=101, y=201, x=61)
    encoding_matrix = SpatialDimension(z=14, y=220, x=61)

    with pytest.raises(NotImplementedError, match='recon_matrix'):
        FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=None, dim=(-4, -2, -1))

    with pytest.raises(NotImplementedError, match='encoding_matrix'):
        FastFourierOp(recon_matrix=None, encoding_matrix=encoding_matrix, dim=(-4, -2, -1))

    
def test_fast_fourier_op_repr():
    """Test the __repr__ method of FastFourierOp."""

    recon_matrix = SpatialDimension(64, 64, 64)
    encoding_matrix = SpatialDimension(128, 128, 128)
    fft_op = FastFourierOp(dim=(-3, -2, -1), recon_matrix=recon_matrix, encoding_matrix=encoding_matrix)
    repr_str = repr(fft_op)

    # Check if __repr__ contains expected information
    assert 'Dimension(s) along which FFT is applied' in repr_str

    
@pytest.mark.cuda
def test_fast_fourier_op_cuda():
    """Test fast Fourier operator works on CUDA devices."""

    # Generate data
    recon_matrix = SpatialDimension(z=101, y=201, x=61)
    encoding_matrix = SpatialDimension(z=14, y=220, x=61)
    generator = RandomGenerator(seed=0)
    x = generator.complex64_tensor(size=(3, 2, recon_matrix.z, recon_matrix.y, recon_matrix.x))

    # Create on CPU, transfer to GPU, run on GPU
    ff_op = FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, dim=(-2, -3))
    ff_op.cuda()
    (y,) = ff_op(x.cuda())
    assert y.is_cuda

    # Create on CPU, run on CPU
    ff_op = FastFourierOp(recon_matrix=recon_matrix, encoding_matrix=encoding_matrix, dim=(-2, -3))
    (y,) = ff_op(x)
    assert y.is_cpu

    # Create on GPU, run on GPU
    ff_op = FastFourierOp(recon_matrix=recon_matrix.cuda(), encoding_matrix=encoding_matrix.cuda(), dim=(-2, -3))
    (y,) = ff_op(x.cuda())
    assert y.is_cuda

    # Create on GPU, transfer to CPU, run on CPU
    ff_op = FastFourierOp(recon_matrix=recon_matrix.cuda(), encoding_matrix=encoding_matrix.cuda(), dim=(-2, -3))
    ff_op.cpu()
    (y,) = ff_op(x)
    assert y.is_cpu
    