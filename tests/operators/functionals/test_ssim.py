"""Test the SSIM functional."""

from collections.abc import Sequence

import pytest
import torch
from mr2.operators.functionals.SSIM import SSIM
from mr2.utils import RandomGenerator


def test_ssim() -> None:
    """Test the SSIM functional."""
    rng = RandomGenerator(0)
    target = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    ssim = SSIM(target)
    perfect = target.clone()
    assert torch.isclose(ssim(perfect)[0], torch.tensor(1.0))
    bad = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    assert ssim(bad)[0] < 0.1


@pytest.mark.parametrize('shape', [(2, 1, 32, 32), (2, 12, 32, 32)])
def test_ssim_mask(shape: Sequence[int]) -> None:
    """Test masking in the SSIM functional"""
    rng = RandomGenerator(0)
    target = rng.float32_tensor(shape, low=0.0, high=1.0)
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[..., 4:-4, 4:-4] = True
    test = rng.rand_like(target) + 0.5 * target + rng.rand_like(target, high=100) * (~mask).float()
    (masked,) = SSIM(target, mask)(test)
    (cropped,) = SSIM(target[..., 4:-4, 4:-4])(test[..., 4:-4, 4:-4])
    torch.testing.assert_close(masked, cropped)
    assert 0.40 < masked.item() < 0.45


def test_ssim_reduction() -> None:
    """Test the reduction argument of the SSIM functional."""
    rng = RandomGenerator(0)
    target = rng.complex64_tensor((2, 3, 10, 10, 10), low=0.0, high=1.0)
    test = rng.complex64_tensor(target.shape) + rng.float32_tensor((2, 3, 1, 1, 1), low=0.2, high=0.8) * target
    (ssim_volume,) = SSIM(target, reduction='volume')(test)
    (ssim_full,) = SSIM(target, reduction='full')(test)
    (ssim_none,) = SSIM(target, reduction='none')(test)
    torch.testing.assert_close(ssim_volume.mean(), ssim_full)
    torch.testing.assert_close(ssim_none.mean(), ssim_full)
    assert ssim_volume.shape == (2, 3)
    assert ssim_full.shape == ()
    assert ssim_none.shape == (2, 3, 4, 4, 4)
