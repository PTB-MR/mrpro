"""Test DropPath."""

import pytest
from mrpro.nn.DropPath import DropPath
from mrpro.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
def test_droppath_no_drop(device):
    """Test DropPath."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5)).to(device)
    droppath = DropPath(0).to(device)
    y = droppath(x)
    assert (y == x).all()


def test_droppath_drop_all():
    """Test DropPath."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5))
    droppath = DropPath(1.0)
    y = droppath(x)
    assert (y == 0).all()
