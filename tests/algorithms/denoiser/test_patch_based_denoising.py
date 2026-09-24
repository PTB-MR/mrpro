"""Tests for patch-based dictionary denoising."""

import pytest
import torch
from mrpro.algorithms.denoiser.patch_based_denoising import patch_based_denoising
from mrpro.data import IData, SpatialDimension
from mrpro.operators import FastFourierOp
from mrpro.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.fixture
def idata_single_coil(ellipse_phantom, random_kheader) -> IData:
    """Create single-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    img = ellipse_phantom.phantom.image_space(image_dimensions)
    return IData.from_tensor_and_kheader(data=img, header=random_kheader)


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_fft_dictionary_no_regularization_is_identity(idata_single_coil: IData, tensor_input: bool) -> None:
    """Without regularization, a unitary dictionary (FFT) returns the input image."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    dictionary_op = FastFourierOp(dim=(-2, -1)).H
    if tensor_input:
        denoised = patch_based_denoising(
            noisy.data,
            dictionary_op,
            patch_dim=(-2, -1),
            patch_size=(8, 8),
            regularization_weight=0.0,
        )
    else:
        denoised = patch_based_denoising(
            noisy,
            dictionary_op,
            patch_dim=(-2, -1),
            patch_size=(8, 8),
            regularization_weight=0.0,
        ).data
    torch.testing.assert_close(denoised, noisy.data)


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_fft_dictionary_denoising(idata_single_coil: IData, tensor_input: bool) -> None:
    """With regularization, sparsity in the FFT dictionary reduces the noise."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    dictionary_op = FastFourierOp(dim=(-2, -1)).H
    if tensor_input:
        denoised = patch_based_denoising(
            noisy.data,
            dictionary_op,
            patch_dim=(-2, -1),
            patch_size=(8, 8),
            regularization_weight=0.1,
        )
    else:
        denoised = patch_based_denoising(
            noisy,
            dictionary_op,
            patch_dim=(-2, -1),
            patch_size=(8, 8),
            regularization_weight=0.1,
        ).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )
