"""Tests for total variation denoising."""

import pytest
from mrpro.algorithms.denoiser.conv_analysis_dictionary_denoising import conv_analysis_dictionary_denoising
from mrpro.data import IData, SpatialDimension
from mrpro.utils import RandomGenerator
from mrpro.utils.filters import dct_filters
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
    dct_kernel = dct_filters(kernel_size=(5, 5))

    if tensor_input:
        denoised = conv_analysis_dictionary_denoising(
            noisy.data,
            kernel=dct_kernel,
            regularization_weight=1e-4,
            max_iterations_pdhg=64,
        )
    else:
        denoised = conv_analysis_dictionary_denoising(
            noisy,
            kernel=dct_kernel,
            regularization_weight=1e-4,
            max_iterations_pdhg=64,
        ).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )
