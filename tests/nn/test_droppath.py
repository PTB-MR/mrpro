"""Test DropPath."""

from mrpro.nn.DropPath import DropPath
from mrpro.utils import RandomGenerator


def test_droppath_no_drop():
    """Test DropPath."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5))
    droppath = DropPath(0)
    y = droppath(x)
    assert (y == x).all()


def test_droppath_drop_all():
    """Test DropPath."""
    rng = RandomGenerator(seed=42)
    x = rng.float32_tensor((1, 3, 4, 5))
    droppath = DropPath(1.0)
    y = droppath(x)
    assert (y == 0).all()
