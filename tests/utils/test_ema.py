"""Tests for EMADict."""

from typing import Any

import pytest
import torch
from mr2.utils import RandomGenerator
from mr2.utils.ema import EMADict


@pytest.mark.parametrize(
    ('key', 'value'),
    [
        ('float', 1.0),
        ('complex', 1.0 + 1.0j),
        ('tensor', torch.ones(2, 3)),
    ],
)
def test_ema_dict_numerical(
    key: str,
    value: Any,
) -> None:
    """Test that EMA calculation is numerically correct."""
    decay = 0.8
    ema = EMADict(decay=decay)

    ema[key] = value
    new_value = RandomGenerator(seed=42).float32() * value
    ema.update({key: new_value})

    expected = decay * value + (1 - decay) * new_value
    if isinstance(value, torch.Tensor):
        torch.testing.assert_close(ema[key], expected)
    else:
        assert ema[key] == pytest.approx(expected)


def test_ema_dict_invalid_decay() -> None:
    """Test EMADict with invalid decay values."""
    with pytest.raises(ValueError, match='Decay must be between 0 and 1'):
        EMADict(decay=-0.1)
    with pytest.raises(ValueError, match='Decay must be between 0 and 1'):
        EMADict(decay=1.1)


def test_ema_dict_update() -> None:
    """Test EMADict update method."""
    rng = RandomGenerator(seed=42)
    ema = EMADict(decay=0.9)

    new_dict: dict[str, Any] = {
        'float': rng.float32(),
        'complex': rng.complex64(),
        'tensor': rng.float32_tensor((2, 3)),
        'string': 'test',
    }
    ema.update(new_dict)

    for key, value in new_dict.items():
        assert key in ema
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(ema[key], value)
        else:
            assert ema[key] == value


def test_ema_dict_deletion() -> None:
    """Test EMADict deletion."""
    rng = RandomGenerator(seed=42)
    ema = EMADict(decay=0.9)

    ema['test'] = rng.float32()
    assert 'test' in ema

    del ema['test']
    assert 'test' not in ema

    with pytest.raises(KeyError):
        del ema['nonexistent']


def test_ema_dict_tensor_detach() -> None:
    """Test that tensors are detached from autograd graph."""
    rng = RandomGenerator(seed=42)
    ema = EMADict(decay=0.9)

    tensor = rng.float32_tensor((2, 3)).requires_grad_(True)
    ema['test'] = tensor
    assert not ema['test'].requires_grad
