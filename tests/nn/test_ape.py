"""Tests for absolute position encoding"""

import pytest
import torch
from mr2.nn import AbsolutePositionEncoding
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', marks=pytest.mark.cuda, id='cuda'),
    ],
)
def test_absolute_position_encodings(device: str) -> None:
    """Test absolute position encoding."""
    n_features = 32
    shape = (1, 2 * n_features, 32, 32)
    ape = AbsolutePositionEncoding(2, n_features, True, 128).to(device)
    rng = RandomGenerator(444)
    x1 = rng.float32_tensor(shape).to(device)
    x2 = rng.float32_tensor(shape).to(device)
    y1, y2 = ape(x1), ape(x2)
    assert y1.shape == x1.shape
    torch.testing.assert_close(y1 - x1, y2 - x2)
    assert (x1[:, n_features:] == y1[:, n_features:]).all()  # unembedded features
