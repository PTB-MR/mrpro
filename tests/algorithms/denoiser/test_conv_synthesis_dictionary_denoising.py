"""Tests for convolutional synthesis dictionary-based image denoising."""

import itertools

import pytest
import torch
from mrpro.algorithms.denoiser.conv_synthesis_dictionary_denoising import conv_synthesis_dictionary_denoising
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
    dct_kernel = dct_filters(kernel_size=(5, 5))[1:]

    if tensor_input:
        denoised = conv_synthesis_dictionary_denoising(
            noisy.data,
            dct_kernel,
            low_pass_parameter=0.1,
            regularization_weight=0.4,
            max_iterations_low_pass_filtering=8,
            max_iterations_pgd=32,
        )
    else:
        denoised = conv_synthesis_dictionary_denoising(
            noisy,
            dct_kernel,
            low_pass_parameter=0.1,
            regularization_weight=0.4,
            max_iterations_low_pass_filtering=8,
            max_iterations_pgd=32,
        ).data
    assert relative_image_difference(denoised, idata_single_coil.data) < relative_image_difference(
        noisy.data, idata_single_coil.data
    )


@pytest.mark.parametrize('kernel_shape', [(9,), (3, 5), (3, 5, 7)])
def test_regularization_parameter_maps(kernel_shape):
    """Test that using regularization parameter maps is possible."""
    rng = RandomGenerator(seed=0)

    n_filters = 8
    kernel = rng.randn_tensor(size=(n_filters, *kernel_shape), dtype=torch.float32)
    noisy = rng.randn_tensor(size=(10, 10, 10), dtype=torch.float32)
    regularization_weight = rng.randn_tensor(size=(kernel.shape[0], 1, 1, 1), dtype=torch.float32)
    _ = conv_synthesis_dictionary_denoising(
        noisy,
        kernel,
        low_pass_parameter=0.1,
        regularization_weight=regularization_weight,
        max_iterations_low_pass_filtering=8,
        max_iterations_pgd=2,
    )


def test_incompatible_regularization_parameter():
    """Test that incompatible regularization parameter tensors throw error."""
    rng = RandomGenerator(seed=0)

    kernel = rng.randn_tensor(size=(8, 5, 5), dtype=torch.float32)
    noisy = rng.randn_tensor(size=(2, 16, 16), dtype=torch.float32)
    regularization_weight = rng.randn_tensor(size=(7, 1, 1, 1), dtype=torch.float32)  # 7!=8

    with pytest.raises(
        ValueError,
        match='must be broadcastable with the output of the convolutional analysis operator',
    ):
        _ = conv_synthesis_dictionary_denoising(
            noisy,
            kernel,
            low_pass_parameter=0.1,
            regularization_weight=regularization_weight,
            max_iterations_low_pass_filtering=8,
            max_iterations_pgd=2,
        )


@pytest.mark.cuda
def test_conv_synthesis_dictionary_denoising_cuda():
    """Test that denoising can be performed on the gpu."""
    rng = RandomGenerator(seed=0)

    kernel = rng.randn_tensor(size=(4, 3, 3), dtype=torch.float32).cuda()
    noisy = rng.randn_tensor(size=(1, 8, 8), dtype=torch.float32).cuda()
    regularization_weight = rng.randn_tensor(size=(4, 1, 1, 1), dtype=torch.float32).cuda()
    denoised = conv_synthesis_dictionary_denoising(
        noisy,
        kernel,
        regularization_weight=regularization_weight,
        low_pass_parameter=0.1,
        max_iterations_low_pass_filtering=1,
        max_iterations_pgd=1,
    )
    assert denoised.is_cuda


@pytest.mark.cuda
@pytest.mark.parametrize(
    ('noisy_device', 'kernel_device', 'weight_device', 'initial_codes_device'),
    itertools.product(('cpu', 'cuda'), repeat=4),
)
def test_different_device_combinations(
    noisy_device: str,
    kernel_device: str,
    weight_device: str,
    initial_codes_device: str,
):
    """Test that the noisy image determines the device to run the algorithm on."""
    rng = RandomGenerator(seed=0)

    kernel = rng.randn_tensor(size=(4, 3, 3), dtype=torch.float32).to(kernel_device)

    noisy = rng.randn_tensor(size=(1, 8, 8), dtype=torch.float32).to(noisy_device)

    regularization_weight = rng.randn_tensor(size=(4, 1, 1), dtype=torch.float32).to(weight_device)

    initial_codes = rng.randn_tensor(size=(4, 1, 8, 8), dtype=torch.float32).to(initial_codes_device)

    denoised = conv_synthesis_dictionary_denoising(
        noisy,
        kernel,
        regularization_weight=regularization_weight,
        low_pass_parameter=0.1,
        initial_codes=initial_codes,
        max_iterations_low_pass_filtering=1,
        max_iterations_pgd=1,
    )

    if noisy.is_cuda:
        assert denoised.is_cuda
    else:
        assert not denoised.is_cuda
