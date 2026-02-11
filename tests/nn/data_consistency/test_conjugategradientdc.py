"""Tests for ConjugateGradientDC module."""

import torch
from mr2.data.KData import KData
from mr2.nn.data_consistency.ConjugateGradientDC import ConjugateGradientDC


def test_conjugate_gradient_dc(
    image_noisy: torch.Tensor, kdata_us: KData, image: torch.Tensor, image_us: torch.Tensor
) -> None:
    image_noisy = image_noisy.clone().requires_grad_(True)
    dc = ConjugateGradientDC(initial_regularization_weight=1.0)
    result = dc(image_noisy, kdata_us)
    loss = (result - image).abs().mean()
    assert loss < (image_noisy - image).abs().mean()
    assert loss < (image_us - image).abs().mean()
    loss.backward()
    assert image_noisy.grad is not None
    assert dc.log_weight.grad is not None
    assert not dc.log_weight.grad.isnan()
    assert not image_noisy.grad.isnan().any()
