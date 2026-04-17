"""Tests for convolutional dictionary operator."""

from typing import Literal

import pytest
import torch
from mrpro.operators import ConvAnalysisDictionaryOp, ConvSynthesisDictionaryOp
from mrpro.utils import RandomGenerator

from tests import (
    dotproduct_adjointness_test,
    forward_mode_autodiff_of_linear_operator_test,
    gradient_of_linear_operator_test,
)


def create_conv_analysis_dictionary_op_and_data(
    img_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> tuple[ConvAnalysisDictionaryOp, torch.Tensor, torch.Tensor]:
    """Create a convolutional analysis dictionary operator and elements from domain and range."""
    rng = RandomGenerator(seed=0)
    rng_image = rng.complex64_tensor if dtype_input == torch.complex64 else rng.float32_tensor

    u = rng_image(size=img_shape)
    v = rng_image(size=(kernel_shape[0], *img_shape))

    rng_kernel = rng.complex64_tensor if dtype_kernel == torch.complex64 else rng.float32_tensor
    kernel = rng_kernel(size=kernel_shape)

    # Generate convolutional dictionary operator
    conv_analysis_dictionary_op = ConvAnalysisDictionaryOp(kernel, pad_mode)

    return conv_analysis_dictionary_op, u, v


def create_conv_synthesis_dictionary_op_and_data(
    img_shape: tuple[int, ...],
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> tuple[ConvSynthesisDictionaryOp, torch.Tensor, torch.Tensor]:
    """Create a convolutional synthesis dictionary operator and elements from domain and range."""
    rng = RandomGenerator(seed=0)
    rng_image = rng.complex64_tensor if dtype_input == torch.complex64 else rng.float32_tensor

    u = rng_image(size=(kernel_shape[0], *img_shape))
    v = rng_image(size=img_shape)

    rng_kernel = rng.complex64_tensor if dtype_kernel == torch.complex64 else rng.float32_tensor
    kernel = rng_kernel(size=kernel_shape)

    # Generate convolutional synthesis dictionary operator
    conv_synthesis_dictionary_op = ConvSynthesisDictionaryOp(kernel, pad_mode)

    return conv_synthesis_dictionary_op, u, v


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_analysis_dictionary_op_adjointness(
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test convolutional analysis dictionary operator adjoint property."""
    img_shape = (2, 3, 1, 8, 10, 11)
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        dotproduct_adjointness_test(
            *create_conv_analysis_dictionary_op_and_data(img_shape, kernel_shape, pad_mode, dtype_input, dtype_kernel)
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_synthesis_dictionary_op_adjointness(
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test convolutional synthesis dictionary operator adjoint property."""
    img_shape = (2, 3, 1, 8, 10, 11)
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        dotproduct_adjointness_test(
            *create_conv_synthesis_dictionary_op_and_data(img_shape, kernel_shape, pad_mode, dtype_input, dtype_kernel)
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_analysis_dictionary_op_forward_mode_autodiff(
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test the forward-mode autodiff of the convolutional dictionary operator."""
    img_shape = (2, 3, 1, 8, 10, 11)
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        forward_mode_autodiff_of_linear_operator_test(
            *create_conv_analysis_dictionary_op_and_data(img_shape, kernel_shape, pad_mode, dtype_input, dtype_kernel),
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4,
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_synthesis_dictionary_op_forward_mode_autodiff(
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test the forward-mode autodiff of the convolutional dictionary operator."""
    img_shape = (2, 3, 1, 8, 10, 11)
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        forward_mode_autodiff_of_linear_operator_test(
            *create_conv_synthesis_dictionary_op_and_data(img_shape, kernel_shape, pad_mode, dtype_input, dtype_kernel),
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4,
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_analysis_dictionary_op_grad(
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test the forward-mode autodiff of the convolutional analysis dictionary operator."""
    img_shape = (2, 3, 8, 9, 10)
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        gradient_of_linear_operator_test(
            *create_conv_analysis_dictionary_op_and_data(img_shape, kernel_shape, pad_mode, dtype_input, dtype_kernel),
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4,
        )


@pytest.mark.parametrize('kernel_shape', [(16, 7), (8, 5, 5), (4, 3, 3, 3)])
@pytest.mark.parametrize('pad_mode', ['replicate', 'constant', 'reflect', 'circular'])
@pytest.mark.parametrize('dtype_input', [torch.complex64, torch.float32])
@pytest.mark.parametrize('dtype_kernel', [torch.complex64, torch.float32])
def test_conv_synthesis_dictionary_op_grad(
    kernel_shape: tuple[int, ...],
    pad_mode: Literal['constant', 'reflect', 'replicate', 'circular'],
    dtype_input: torch.dtype,
    dtype_kernel: torch.dtype,
) -> None:
    """Test the forward-mode autodiff of the convolutional synthesis dictionary operator."""
    img_shape = (2, 3, 8, 9, 10)
    if not (dtype_input == torch.float32 and dtype_kernel == torch.complex64):
        gradient_of_linear_operator_test(
            *create_conv_synthesis_dictionary_op_and_data(img_shape, kernel_shape, pad_mode, dtype_input, dtype_kernel),
            relative_tolerance=1e-4,
            absolute_tolerance=1e-4,
        )


@pytest.mark.parametrize('kernel_shape', [(4, 3, 3), (8, 5, 5, 5)])
def test_conv_dictionary_wrong_input_shape(kernel_shape: tuple[int, ...]):
    """Test that an error is raised if the inputs dimension is not compatible with the kernel."""
    rng = RandomGenerator(seed=0)

    input_tensor = rng.float32_tensor(size=(kernel_shape[0] + 1, 16, 15, 14))

    kernel = rng.float32_tensor(size=kernel_shape)
    conv_op_synthesis = ConvSynthesisDictionaryOp(kernel, 'circular')
    conv_op_analysis = ConvAnalysisDictionaryOp(kernel, 'circular')

    with pytest.raises(ValueError, match='First dimension of input must match'):
        conv_op_synthesis(input_tensor)
    with pytest.raises(ValueError, match='First dimension of input must match'):
        conv_op_analysis.H(input_tensor)


def test_conv_dictionary_wrong_pad_mode():
    """Test that an error is raised if the padding mode is not valid."""
    rng = RandomGenerator(seed=0)
    kernel = rng.float32_tensor(size=(4, 3, 3))
    with pytest.raises(ValueError, match='Pad mode must be either'):
        _ = ConvAnalysisDictionaryOp(kernel, 'my_pad_mode')  # type: ignore[arg-type]
    with pytest.raises(ValueError, match='Pad mode must be either'):
        _ = ConvSynthesisDictionaryOp(kernel, 'my_pad_mode')  # type: ignore[arg-type]


def test_conv_dictionary_invalid_dtype_combination():
    """Test that an error is raised a real-valued input is used with a complex-valued kernel."""
    img_shape = (4, 32, 32)
    kernel_shape = (4, 3, 3)
    dtype_kernel = torch.complex64
    dtype_input = torch.float32
    conv_op_synthesis, u_synthesis, v_synthesis = create_conv_synthesis_dictionary_op_and_data(
        img_shape, kernel_shape, 'circular', dtype_input, dtype_kernel
    )
    conv_op_analysis, u_analysis, v_analysis = create_conv_analysis_dictionary_op_and_data(
        img_shape, kernel_shape, 'circular', dtype_input, dtype_kernel
    )
    with pytest.raises(ValueError, match='Input tensor must be complex-valued'):
        conv_op_synthesis(u_synthesis)
    with pytest.raises(ValueError, match='Input tensor must be complex-valued'):
        conv_op_synthesis.H(v_synthesis)
    with pytest.raises(ValueError, match='Input tensor must be complex-valued'):
        conv_op_analysis(u_analysis)
    with pytest.raises(ValueError, match='Input tensor must be complex-valued'):
        conv_op_analysis.H(v_analysis)


@pytest.mark.cuda
def test_conv_analysis_dictionary_op_cuda() -> None:
    """Test convolutional dictionary operator works on CUDA devices."""
    random_generator = RandomGenerator(seed=0)
    kernel = random_generator.complex64_tensor(size=(4, 3, 3))
    im_shape = (4, 8, 10)
    u = random_generator.complex64_tensor(size=im_shape)

    # Create on CPU, run on CPU
    conv_dictionary_op = ConvAnalysisDictionaryOp(kernel, pad_mode='circular')
    operator = conv_dictionary_op.H @ conv_dictionary_op
    (conv_dictionary_output,) = operator(u)
    assert conv_dictionary_output.is_cpu

    # Transfer to GPU, run on GPU
    conv_dictionary_op = ConvAnalysisDictionaryOp(kernel, pad_mode='circular')
    operator = conv_dictionary_op.H @ conv_dictionary_op
    operator.cuda()
    (conv_dictionary_output,) = operator(u.cuda())
    assert conv_dictionary_output.is_cuda


@pytest.mark.cuda
def test_conv_synthesis_dictionary_op_cuda() -> None:
    """Test convolutional dictionary operator works on CUDA devices."""
    random_generator = RandomGenerator(seed=0)
    kernel = random_generator.complex64_tensor(size=(4, 3, 3))
    im_shape = (kernel.shape[0], 4, 8, 10)
    u = random_generator.complex64_tensor(size=im_shape)

    # Create on CPU, run on CPU
    conv_dictionary_op = ConvSynthesisDictionaryOp(kernel, pad_mode='circular')
    operator = conv_dictionary_op.H @ conv_dictionary_op
    (conv_dictionary_output,) = operator(u)
    assert conv_dictionary_output.is_cpu

    # Transfer to GPU, run on GPU
    conv_dictionary_op = ConvSynthesisDictionaryOp(kernel, pad_mode='circular')
    operator = conv_dictionary_op.H @ conv_dictionary_op
    operator.cuda()
    (conv_dictionary_output,) = operator(u.cuda())
    assert conv_dictionary_output.is_cuda
