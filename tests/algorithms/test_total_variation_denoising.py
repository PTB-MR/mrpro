"""Tests for total variation denoising."""

import pytest
from mrpro.algorithms.total_variation_denoising import total_variation_denoising
from mrpro.data import IData, SpatialDimension
from mrpro.utils import RandomGenerator
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
        denoised = total_variation_denoising(noisy.data, regularization_weights=[1.0, 1.0])
    else:
        denoised = total_variation_denoising(noisy, regularization_weights=[1.0, 1.0]).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )
