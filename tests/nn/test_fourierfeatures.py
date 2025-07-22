"""Test for random fourier features"""

import pytest
from mrpro.nn import FourierFeatures
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_fourierfeatures(device) -> None:
    """Test fourier features"""
    n_features_in = 1
    n_features_out = 16
    std = 1.0
    rng = RandomGenerator(444)
    x = rng.float32_tensor((1, n_features_in)).to(device)
    ff = FourierFeatures(n_features_in, n_features_out, std).to(device)
    y = ff(x)
    assert y.shape == (1, n_features_out)
