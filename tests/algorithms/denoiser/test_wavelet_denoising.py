"""Tests for wavelet denoising."""

import pytest
import torch
from mrpro.algorithms.denoiser.wavelet_denoising import wavelet_denoising
from mrpro.data import IData, SpatialDimension
from mrpro.operators.WaveletOp import WaveletType
from mrpro.utils import RandomGenerator
from tests.helper import relative_image_difference


@pytest.fixture
def idata_single_coil(ellipse_phantom, random_kheader) -> IData:
    """Create single-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ellipse_phantom.n_y, x=ellipse_phantom.n_x)
    img = ellipse_phantom.phantom.image_space(image_dimensions)
    return IData.from_tensor_and_kheader(data=img, header=random_kheader)


@pytest.mark.parametrize('tensor_input', [True, False], ids=['tensor', 'idata'])
def test_wavelet_denoising(idata_single_coil: IData, tensor_input: bool) -> None:
    """Denoising has to reduce the difference to the noise-free image."""
    rng = RandomGenerator(seed=0)
    noisy = IData(idata_single_coil.data + rng.rand_like(idata_single_coil.data), idata_single_coil.header)
    if tensor_input:
        denoised = wavelet_denoising(noisy.data, regularization_dim=(-2, -1), regularization_weight=0.1)
    else:
        denoised = wavelet_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=0.1).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )


@pytest.mark.parametrize('wavelet_name', ['haar', 'db4', 'sym4'])
def test_wavelet_denoising_wavelets(idata_single_coil: IData, wavelet_name: WaveletType) -> None:
    """Denoising works for different wavelets and a fixed number of levels."""
    rng = RandomGenerator(seed=0)
    noisy = idata_single_coil.data + rng.rand_like(idata_single_coil.data)
    denoised = wavelet_denoising(
        noisy, regularization_dim=(-2, -1), regularization_weight=0.1, wavelet_name=wavelet_name, level=2
    )
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy, idata_single_coil.data
    )


def test_wavelet_denoising_complex(idata_single_coil: IData) -> None:
    """Complex input stays complex and the phase is denoised as well."""
    rng = RandomGenerator(seed=0)
    img = idata_single_coil.data * torch.exp(1j * torch.linspace(-1, 1, idata_single_coil.data.shape[-1]))
    noisy = img + 0.1 * rng.complex64_tensor(img.shape)
    denoised = wavelet_denoising(noisy, regularization_dim=(-2, -1), regularization_weight=0.1)
    assert denoised.is_complex()
    assert relative_image_difference(denoised, img) < relative_image_difference(noisy, img)


def test_wavelet_denoising_zero_weight(idata_single_coil: IData) -> None:
    """A weight of zero has to leave the image unchanged, as the wavelet transform is orthonormal."""
    denoised = wavelet_denoising(idata_single_coil.data, regularization_dim=(-2, -1), regularization_weight=0.0)
    torch.testing.assert_close(denoised, idata_single_coil.data, atol=1e-5, rtol=1e-4)


def test_wavelet_denoising_large_weight(idata_single_coil: IData) -> None:
    """A very large weight removes everything."""
    denoised = wavelet_denoising(idata_single_coil.data, regularization_dim=(-2, -1), regularization_weight=1e6)
    assert denoised.abs().sum() == 0


def test_wavelet_denoising_repeated_dims() -> None:
    """Error for repeated dims."""
    with pytest.raises(ValueError, match='Repeated values are not allowed in regularization_dim'):
        _ = wavelet_denoising(torch.zeros(4, 4), regularization_dim=(-2, 0), regularization_weight=0.1)
