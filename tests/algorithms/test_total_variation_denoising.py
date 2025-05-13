"""Tests for total variation denoising."""

import torch
from mrpro.algorithms.total_variation_denoising import total_variation_denoising
from mrpro.data import IData, SpatialDimension
from mrpro.utils import RandomGenerator
from tests.helper import relative_image_difference


def idata_single_coil(ph_ellipse, random_kheader):
    """Create single-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ph_ellipse.n_y, x=ph_ellipse.n_x)
    img = ph_ellipse.phantom.image_space(image_dimensions)
    return IData.from_tensor_and_kheader(data=img, header=random_kheader)


def test_denoising(ellipse_phantom, random_kheader):
    """Test total variation denoising."""
    idata = idata_single_coil(ellipse_phantom, random_kheader)
    rng = RandomGenerator(seed=0)
    noisy_idata = IData(idata.data + rng.complex64_tensor(idata.data.shape), idata.header)
    # denoising of IData
    denoised_idata = total_variation_denoising(noisy_idata, regularization_weights=[1.0, 1.0])
    # denoising of tensor
    denoised_tensor = total_variation_denoising(noisy_idata.data, regularization_weights=[1.0, 1.0])
    assert relative_image_difference(idata.data, denoised_idata.data) < relative_image_difference(
        idata.data, noisy_idata.data
    )
    torch.testing.assert_close(denoised_idata.data, denoised_tensor, atol=1e-3, rtol=1e-3)
