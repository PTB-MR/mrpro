"""Test the SSIM functional."""

import torch
from mrpro.operators.functionals.SSIM import SSIM
from mrpro.utils import RandomGenerator


def test_ssim() -> None:
    """Test the SSIM functional."""
    rng = RandomGenerator(0)
    target = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    ssim = SSIM(target)
    perfect = target.clone()
    assert torch.isclose(ssim(perfect)[0], torch.tensor(1.0))
    bad = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    assert ssim(bad)[0] < 0.1


def test_ssim_mask() -> None:
    """Test masking in the SSIM functional"""
    rng = RandomGenerator(0)
    target = rng.float32_tensor((1, 8, 32, 32), low=0.0, high=1.0)
    mask = torch.zeros(1, 8, 32, 32, dtype=torch.bool)
    mask[..., 4:-4, 4:-4] = True
    test = rng.rand_like(target) + 0.5 * target + rng.rand_like(target, high=100) * (~mask).float()
    (masked,) = SSIM(target, mask)(test)
    (cropped,) = SSIM(target[..., 4:-4, 4:-4])(test[..., 4:-4, 4:-4])
    torch.testing.assert_close(masked, cropped)
    assert 0.4 < masked.item() < 0.6
