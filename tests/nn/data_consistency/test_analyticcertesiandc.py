import torch
from mrpro.data.KData import KData
from mrpro.nn.data_consistency.AnalyticCartesianDC import AnalyticCartesianDC


def test_analytic_cartesian_dc(image_noisy: torch.Tensor, kdata_us: KData, image: torch.Tensor, image_us: torch.Tensor):
    image_noisy = image_noisy.clone().requires_grad_(True)
    dc = AnalyticCartesianDC(initial_regularization_weight=1e-6)
    result = dc(image_noisy, kdata_us)
    loss = (result - image).abs().mean()
    assert loss < (image_noisy - image).abs().mean()
    assert loss < (image_us - image).abs().mean()
    loss.backward()
    assert image_noisy.grad is not None
    assert dc.log_weight.grad is not None
    assert not dc.log_weight.grad.isnan()
    assert not image_noisy.grad.isnan().any()
