"""Tests for domain thresholding."""

from collections.abc import Callable

import pytest
import torch
from mrpro.algorithms.denoiser import transform_domain_soft_thresholding
from mrpro.data import IData
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.WaveletOp import WaveletOp
from mrpro.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_transform_domain_soft_thresholding(idata_single_coil: IData, tensor_input: bool) -> None:
    """Denoising has to reduce the difference to the noise-free image."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    transform_op = WaveletOp(domain_shape=noisy.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)

    if tensor_input:
        denoised = transform_domain_soft_thresholding(noisy.data, transform_op, threshold=0.1)
    else:
        denoised = transform_domain_soft_thresholding(noisy, transform_op, threshold=0.1).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )


def _wavelet_op(shape: torch.Size) -> LinearOperator:
    return WaveletOp(domain_shape=shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)


def _fourier_op(shape: torch.Size) -> LinearOperator:
    return FastFourierOp(dim=(-2, -1))


@pytest.mark.parametrize('make_transform_op', [_wavelet_op, _fourier_op], ids=['wavelet', 'fourier'])
def test_transform_domain_soft_thresholding_denoises(
    idata_single_coil: IData,
    make_transform_op: Callable[[torch.Size], LinearOperator],
) -> None:
    """Thresholding in a transform domain reduces the error w.r.t. the clean image."""
    rng = RandomGenerator(seed=0)
    clean = idata_single_coil.data
    noisy = clean + rng.rand_like(clean)

    denoised = transform_domain_soft_thresholding(noisy, make_transform_op(noisy.shape), threshold=0.1)

    assert denoised.shape == noisy.shape
    assert relative_image_difference(denoised, clean) < relative_image_difference(noisy, clean)


def test_transform_domain_soft_thresholding_zero_weight(idata_single_coil: IData) -> None:
    """A weight of zero has to leave the image unchanged, if the transform is orthonormal."""
    transform_op = WaveletOp(domain_shape=idata_single_coil.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)
    denoised = transform_domain_soft_thresholding(idata_single_coil.data, transform_op, threshold=0.0)
    torch.testing.assert_close(denoised, idata_single_coil.data, atol=1e-5, rtol=1e-4)


def test_transform_domain_soft_thresholding_large_weight(idata_single_coil: IData) -> None:
    """A very large weight removes everything."""
    transform_op = WaveletOp(domain_shape=idata_single_coil.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)
    denoised = transform_domain_soft_thresholding(idata_single_coil.data, transform_op, threshold=1e6)
    assert denoised.abs().sum() == 0


@pytest.mark.cuda
def test_transform_domain_soft_thresholding_cuda(idata_single_coil: IData) -> None:
    """Test denoising on CUDA device."""
    transform_op = WaveletOp(domain_shape=idata_single_coil.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)
    denoised = transform_domain_soft_thresholding(idata_single_coil.cuda(), transform_op, threshold=0.1)
    assert denoised.is_cuda
