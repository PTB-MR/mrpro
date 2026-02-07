"""Tests for FiLM module."""

from collections.abc import Sequence

import pytest
from mrpro.nn.FiLM import FiLM
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('n_channels', 'n_channels_cond', 'input_shape', 'cond_shape'),
    [
        (64, 32, (1, 64, 32, 32), (1, 32)),
        (32, 16, (2, 32, 16, 16), (2, 16)),
    ],
)
def test_film(
    n_channels: int, n_channels_cond: int, input_shape: Sequence[int], cond_shape: Sequence[int], device: str
) -> None:
    """Test FiLM output shape and backpropagation."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor(input_shape).to(device).requires_grad_(True)
    cond = rng.float32_tensor(cond_shape).to(device).requires_grad_(True)
    film = FiLM(channels=n_channels, cond_dim=n_channels_cond).to(device)
    output = film(x, cond=cond)
    assert output.shape == x.shape, f'Output shape {output.shape} != input shape {x.shape}'
    output.sum().backward()
    assert x.grad is not None, 'No gradient computed for input'
    assert cond.grad is not None, 'No gradient computed for conditioning'
    assert not output.isnan().any(), 'NaN values in output'
    assert not cond.isnan().any(), 'NaN values in conditioning'
    assert not x.grad.isnan().any(), 'NaN values in input gradients'
    assert not cond.grad.isnan().any(), 'NaN values in conditioning gradients'
    assert film.project is not None, 'Linear layer is not initialized'
    assert next(film.project.parameters()).grad is not None, 'No gradient computed for Linear layer'
