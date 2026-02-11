"""Tests for FiLM module."""

from collections.abc import Sequence

import pytest
import torch
from mr2.nn.FiLM import FiLM
from mr2.utils import RandomGenerator


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


def test_film_features_last() -> None:
    """Test FiLM with features_last=True vs features_last=False."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5))
    cond = rng.float32_tensor((1, 8))

    film_last = FiLM(channels=3, cond_dim=8, features_last=True)
    film = FiLM(channels=3, cond_dim=8, features_last=False)
    film.load_state_dict(film_last.state_dict())

    y_last = film_last(x.moveaxis(1, -1), cond=cond)
    y = film(x, cond=cond)
    torch.testing.assert_close(y, y_last.moveaxis(-1, 1))
