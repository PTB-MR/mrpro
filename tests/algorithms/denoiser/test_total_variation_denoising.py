"""Tests for total variation denoising."""

import pytest
import torch
from mrpro.algorithms.denoiser.total_variation_denoising import total_variation_denoising
from mrpro.data import IData
from mrpro.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_denoising(idata_single_coil: IData, tensor_input: bool) -> None:
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    if tensor_input:
        denoised = total_variation_denoising(noisy.data, regularization_dim=(-2, -1), regularization_weight=[1.0, 1.0])
    else:
        denoised = total_variation_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=[1.0, 1.0]).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )


def test_denoising_same_weight_for_all_dims(idata_single_coil: IData) -> None:
    """Test denoising with same weight for all dims."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    denoised_same_weight = total_variation_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=1.0)
    denoised = total_variation_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=(1.0, 1.0))
    assert relative_image_difference(denoised.data, denoised_same_weight.data) < 1e-2


def test_denoising_weight_dim_mismatch() -> None:
    """Error for different length of dim and weights."""
    with pytest.raises(ValueError, match='Regularization dimensions and weights must have the same length'):
        _ = total_variation_denoising(torch.zeros(1, 1), regularization_dim=(-2,), regularization_weight=[1.0, 1.0])


def test_denoising_repeated_dims() -> None:
    """Error for repeated dims."""
    with pytest.raises(ValueError, match='Repeated values are not allowed in regularization_dim'):
        _ = total_variation_denoising(torch.zeros(1, 1), regularization_dim=(-2, 0), regularization_weight=[1.0, 1.0])


@pytest.mark.cuda
def test_direct_reconstruction_cuda_from_kdata(idata_single_coil: IData) -> None:
    """Test denoising on CUDA device."""
    denoised = total_variation_denoising(
        idata_single_coil.cuda(), regularization_dim=(-2, -1), regularization_weight=[1.0, 1.0], max_iterations=2
    )
    assert denoised.is_cuda
