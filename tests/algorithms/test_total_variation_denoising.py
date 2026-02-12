"""Tests for total variation denoising."""

import pytest
import torch
from mr2.algorithms.total_variation_denoising import total_variation_denoising
from mr2.data import IData, SpatialDimension
from mr2.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.fixture
def idata_single_coil(ellipse_phantom, random_kheader) -> IData:
    """Create single-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    img = ellipse_phantom.phantom.image_space(image_dimensions)
    return IData.from_tensor_and_kheader(data=img, header=random_kheader)


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
