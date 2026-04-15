"""Tests for convolutional dictionary operator."""

from typing import Literal

import pytest
import torch
from mrpro.operators import ConvDictionaryOp
from mrpro.utils import RandomGenerator

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_conv_dictionary_op_and_data(
    kernel_shape: tuple[int, ...],
    mode: Literal['analysis', 'synthesis'],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> tuple[ConvDictionaryOp, torch.Tensor, torch.Tensor]:
    """Create a convolutional dictionary operator and elements from domain and range."""
    rng = RandomGenerator(seed=0)
    img_shape = (2, 3, 1, 8, 10, 11)
    rng_image = rng.complex64_tensor if dtype_input == torch.complex64 else rng.float32_tensor

    if mode == 'synthesis':
        u = rng_image(size=(kernel_shape[0], *img_shape))
        v = rng_image(size=img_shape)
    else:
        u = rng_image(size=img_shape)
        v = rng_image(size=(kernel_shape[0], *img_shape))

    rng_kernel = rng.complex64_tensor if dtype_kernel == torch.complex64 else rng.float32_tensor
    kernel = rng_kernel(size=kernel_shape)

    # Generate convolutional dictionary operator
    conv_dictionary_op = ConvDictionaryOp(kernel, mode, pad_mode)

    return conv_dictionary_op, u, v


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('mode', ['analysis', 'synthesis'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_dictionary_op_adjointness(
    kernel_shape: tuple[int, ...],
    mode: Literal['analysis', 'synthesis'],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test convolutional dictionary operator adjoint property."""
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        dotproduct_adjointness_test(
            *create_conv_dictionary_op_and_data(kernel_shape, mode, pad_mode, dtype_input, dtype_kernel)
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('mode', ['analysis', 'synthesis'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_dictionary_op_forward_mode_autodiff(
    kernel_shape: tuple[int, ...],
    mode: Literal['analysis', 'synthesis'],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test the forward-mode autodiff of the convolutional dictionary operator."""
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        forward_mode_autodiff_of_linear_operator_test(
            *create_conv_dictionary_op_and_data(kernel_shape, mode, pad_mode, dtype_input, dtype_kernel),
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4,
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('mode', ['analysis', 'synthesis'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_dictionary_op_grad(
    kernel_shape: tuple[int, ...],
    mode: Literal['analysis', 'synthesis'],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test the forward-mode autodiff of the convolutional dictionary operator."""
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        gradient_of_linear_operator_test(
            *create_conv_dictionary_op_and_data(kernel_shape, mode, pad_mode, dtype_input, dtype_kernel),
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4,
        )


@pytest.mark.parametrize('kernel_shape', [(32, 6), (16, 8, 9), (8, 6, 5, 5)])
def test_conv_dictionary_even_shape(kernel_shape: tuple[int, ...]):
    """Test that an error is raised if any of the convolutional kernel dimensions is even."""
    rng = RandomGenerator(seed=0)
    kernel = rng.float32_tensor(size=kernel_shape)
    with pytest.raises(ValueError, match='kernel must be odd'):
        _ = ConvDictionaryOp(kernel, 'analysis', 'circular')


@pytest.mark.parametrize('kernel_shape', [(4, 3, 3, 3, 3)])
def test_conv_dictionary_too_many_dimensions(kernel_shape: tuple[int, ...]):
    """Test that an error is raised if the dimension of the kernels is greater than three."""
    rng = RandomGenerator(seed=0)
    kernel = rng.float32_tensor(size=kernel_shape)
    with pytest.raises(ValueError, match='Only 1D, 2D and 3D filters are supported'):
        _ = ConvDictionaryOp(kernel, 'analysis', 'circular')


@pytest.mark.parametrize('kernel_shape', [(4, 3, 3), (8, 5, 5, 5)])
def test_conv_dictionary_wrong_input_shape(kernel_shape: tuple[int, ...]):
    """Test that an error is raised if the inputs dimension is not compatible with the kernel."""
    rng = RandomGenerator(seed=0)

    input_tensor = rng.float32_tensor(size=(kernel_shape[0] + 1, 16, 15, 14))

    kernel = rng.float32_tensor(size=kernel_shape)
    conv_op_synthesis = ConvDictionaryOp(kernel, 'synthesis', 'circular')
    conv_op_analysis = ConvDictionaryOp(kernel, 'analysis', 'circular')

    with pytest.raises(ValueError, match='First dimension of input must match'):
        conv_op_synthesis(input_tensor)
    with pytest.raises(ValueError, match='First dimension of input must match'):
        conv_op_analysis.H(input_tensor)


@pytest.mark.parametrize('mode', ['analsis', 'snthesis'])
def test_conv_dictionary_wrong_mode(mode: Literal['analysis', 'synthesis']):
    """Test that an error is raised if the mode is not "analysis" or "synthesis"."""
    rng = RandomGenerator(seed=0)

    kernel = rng.float32_tensor(size=(4, 3, 3))
    with pytest.raises(ValueError, match='Mode must be either'):
        _ = ConvDictionaryOp(kernel, mode, 'circular')


def test_conv_dictionary_wrong_pad_mode():
    """Test that an error is raised if the padding mode is not valid."""
    rng = RandomGenerator(seed=0)
    kernel = rng.float32_tensor(size=(4, 3, 3))
    with pytest.raises(ValueError, match='Pad mode must be either'):
        _ = ConvDictionaryOp(kernel, 'analysis', 'my_pad_mode')  # type: ignore[arg-type]


@pytest.mark.parametrize('mode', ['analysis', 'synthesis'])
def test_conv_dictionary_invalid_dtype_combination(mode: Literal['analysis', 'synthesis']):
    """Test that an error is raised a real-valued input is used with a complex-valued kernel."""
    kernel_shape = (4, 3, 3)
    dtype_kernel = torch.complex64
    dtype_input = torch.float32
    conv_op, u, v = create_conv_dictionary_op_and_data(kernel_shape, mode, 'circular', dtype_input, dtype_kernel)
    with pytest.raises(ValueError, match='Input tensor must be complex-valued'):
        conv_op(u)
    with pytest.raises(ValueError, match='Input tensor must be complex-valued'):
        conv_op.H(v)


@pytest.mark.parametrize('mode', ['analysis', 'synthesis'])
@pytest.mark.cuda
def test_conv_dictionary_op_cuda(mode: Literal['analysis', 'synthesis']) -> None:
    """Test convolutional dictionary operator works on CUDA devices."""
    random_generator = RandomGenerator(seed=0)
    kernel = random_generator.complex64_tensor(size=(4, 3, 3))
    im_shape = (kernel.shape[0], 4, 8, 10) if mode == 'synthesis' else (4, 8, 10)
    u = random_generator.complex64_tensor(size=im_shape)

    # Create on CPU, run on CPU
    conv_dictionary_op = ConvDictionaryOp(kernel, mode=mode, pad_mode='circular')
    operator = conv_dictionary_op.H @ conv_dictionary_op
    (finite_difference_output,) = operator(u)
    assert finite_difference_output.is_cpu

    # Transfer to GPU, run on GPU
    conv_dictionary_op = ConvDictionaryOp(kernel, mode=mode, pad_mode='circular')
    operator = conv_dictionary_op.H @ conv_dictionary_op
    operator.cuda()
    (finite_difference_output,) = operator(u.cuda())
    assert finite_difference_output.is_cuda
