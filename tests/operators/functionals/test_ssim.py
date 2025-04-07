"""Test the SSIM functional."""

import torch
from mrpro.operators.functionals.SSIM import SSIM

from tests import RandomGenerator
import pytest


def test_ssim() -> None:
    """Test the SSIM functional."""
    rng = RandomGenerator(seed=0)
    target = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    target[..., 2:6, 20:-20, 20:-20] = 0.5
    ssim = SSIM(target)
    perfect = target.clone()
    assert torch.isclose(ssim(perfect)[0], torch.tensor(1.0))
    bad = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    assert ssim(bad)[0] < 0.1
