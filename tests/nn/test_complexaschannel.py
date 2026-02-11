"""Tests for ComplexAsChannel module."""

import pytest
from mr2.nn.ComplexAsChannel import ComplexAsChannel
from mr2.utils import RandomGenerator
from torch.nn import Linear


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
def test_complexaschannel(device: str) -> None:
    """Test ComplexAsChannel output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    input_shape = (1, 32)
    x = rng.complex64_tensor(input_shape).to(device).requires_grad_(True)
    module = ComplexAsChannel(Linear(input_shape[1] * 2, input_shape[1] * 2)).to(device)
    output = module(x)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    assert output.is_complex(), 'Output is not complex'
    output.sum().abs().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert not output.isnan().any(), 'NaN values in output'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert module.module.weight.grad is not None, 'No gradient computed for weight'
    assert module.module.bias.grad is not None, 'No gradient computed for bias'
