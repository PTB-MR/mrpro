"""Tests for total variation denoising."""

import pytest
from mrpro.algorithms.denoiser.conv_synthesis_dictionary_denoising import conv_synthesis_dictionary_denoising
from mrpro.data import IData, SpatialDimension
from mrpro.utils import RandomGenerator
from mrpro.utils.filters import gabor_filters_2d
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
    noisy = IData(
        idata_single_coil.data + rng.rand_like(idata_single_coil.data),
        idata_single_coil.header,
    )
    gabor_kernel = gabor_filters_2d(n_filters=16, kernel_size=(5, 5))

    if tensor_input:
        denoised = conv_synthesis_dictionary_denoising(
            noisy.data,
            gabor_kernel,
            low_pass_parameter=0.1,
            regularization_weight=0.4,
            max_iterations_low_pass_filtering=8,
            max_iterations_pgd=32,
        )
    else:
        denoised = conv_synthesis_dictionary_denoising(
            noisy,
            gabor_kernel,
            low_pass_parameter=0.1,
            regularization_weight=0.4,
            max_iterations_low_pass_filtering=8,
            max_iterations_pgd=32,
        ).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )
