"""Tests for GradientDescentDC module."""

import torch
from mr2.data.KData import KData
from mr2.nn.data_consistency.GradientDescentDC import GradientDescentDC


def test_gradient_descent_dc(
    image_noisy: torch.Tensor, kdata_us: KData, image: torch.Tensor, image_us: torch.Tensor
) -> None:
    image_noisy = image_noisy.clone().requires_grad_(True)
    dc = GradientDescentDC(initial_stepsize=1.0)
    result = dc(image_noisy, kdata_us)
    loss = (result - image).abs().mean()
    assert loss < (image_noisy - image).abs().mean()
    assert loss < (image_us - image).abs().mean()
    loss.backward()
    assert image_noisy.grad is not None
    assert dc.log_stepsize.grad is not None
    assert not dc.log_stepsize.grad.isnan()
    assert not image_noisy.grad.isnan().any()
