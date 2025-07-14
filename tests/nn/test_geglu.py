"""Test GEGLU."""

import torch
from mrpro.nn.GEGLU import GEGLU
from mrpro.utils import RandomGenerator


def test_geglu():
    """Test GELU."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).requires_grad_(True)
    gelu = GEGLU(3, 4)
    y = gelu(x)
    assert y.shape == (1, 4, 4, 5)

    y.sum().backward()
    assert x.grad is not None
    assert gelu.proj.weight.grad is not None


def test_geglu_features_last():
    """Test GELU with features last."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).requires_grad_(True)
    gelu_last = GEGLU(3, 4, features_last=True)
    gelu = GEGLU(3, 4, features_last=False)
    gelu.proj = gelu_last.proj  # need to set the same weights
    y_last = gelu_last(x.moveaxis(1, -1))
    y = gelu(x)
    torch.testing.assert_close(y, y_last.moveaxis(-1, 1))
