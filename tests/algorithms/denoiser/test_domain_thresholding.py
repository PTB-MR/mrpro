"""Tests for domain thresholding."""

from collections.abc import Callable

import pytest
import torch
from mrpro.algorithms.denoiser import domain_thresholding
from mrpro.data import IData
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.WaveletOp import WaveletOp
from mrpro.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_domain_thresholding(idata_single_coil: IData, tensor_input: bool) -> None:
    """Denoising has to reduce the difference to the noise-free image."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    domain_op = WaveletOp(domain_shape=noisy.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)

    if tensor_input:
        denoised = domain_thresholding(noisy.data, domain_op, torch.nn.functional.softshrink, threshold=0.1)
    else:
        denoised = domain_thresholding(noisy, domain_op, torch.nn.functional.softshrink, threshold=0.1).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )


def _wavelet_op(shape: torch.Size) -> LinearOperator:
    return WaveletOp(domain_shape=shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)


def _fourier_op(shape: torch.Size) -> LinearOperator:
    return FastFourierOp(dim=(-2, -1))


@pytest.mark.parametrize('make_domain_op', [_wavelet_op, _fourier_op], ids=['wavelet', 'fourier'])
@pytest.mark.parametrize(
    'threshold_function',
    [torch.nn.functional.softshrink, torch.nn.functional.hardshrink],
    ids=['soft', 'hard'],
)
def test_domain_thresholding_denoises(
    idata_single_coil: IData,
    make_domain_op: Callable[[torch.Size], LinearOperator],
    threshold_function: Callable[..., torch.Tensor],
) -> None:
    """Thresholding in a transform domain reduces the error w.r.t. the clean image."""
    rng = RandomGenerator(seed=0)
    clean = idata_single_coil.data
    noisy = clean + rng.rand_like(clean)

    denoised = domain_thresholding(noisy, make_domain_op(noisy.shape), threshold_function, threshold=0.1)

    assert denoised.shape == noisy.shape
    assert relative_image_difference(denoised, clean) < relative_image_difference(noisy, clean)


def test_domain_thresholding_zero_weight(idata_single_coil: IData) -> None:
    """A weight of zero has to leave the image unchanged, as the wavelet transform is orthonormal."""
    domain_op = WaveletOp(domain_shape=idata_single_coil.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)
    denoised = domain_thresholding(idata_single_coil.data, domain_op, torch.nn.functional.softshrink, threshold=0.0)
    torch.testing.assert_close(denoised, idata_single_coil.data, atol=1e-5, rtol=1e-4)


def test_domain_thresholding_large_weight(idata_single_coil: IData) -> None:
    """A very large weight removes everything."""
    domain_op = WaveletOp(domain_shape=idata_single_coil.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)
    denoised = domain_thresholding(idata_single_coil.data, domain_op, torch.nn.functional.softshrink, threshold=1e6)
    assert denoised.abs().sum() == 0


@pytest.mark.cuda
def test_domain_thresholding_cuda(idata_single_coil: IData) -> None:
    """Test denoising on CUDA device."""
    domain_op = WaveletOp(domain_shape=idata_single_coil.shape[-2:], dim=(-2, -1), wavelet_name='haar', level=2)
    denoised = domain_thresholding(idata_single_coil.cuda(), domain_op, torch.nn.functional.softshrink, threshold=0.1)
    assert denoised.is_cuda
