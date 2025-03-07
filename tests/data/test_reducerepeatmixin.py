"""Test the ReduceRepeatMixin."""

import dataclasses

import torch
from mrpro.data import ReduceRepeatMixin, Rotation, SpatialDimension

from tests import RandomGenerator


@dataclasses.dataclass
class Dummy(ReduceRepeatMixin):
    a: torch.Tensor
    b: SpatialDimension
    c: Rotation

    def __post_init__(self) -> None:
        self.a.ravel()[0] = 1.0


def test_reducerepeatmixin() -> None:
    """Test ReduceRepeatMixin."""
    rng = RandomGenerator(10)

    a = rng.float32_tensor((5, 1, 1, 1))
    a_expanded = a.expand(5, 2, 3, 1)

    b = SpatialDimension(*rng.float32_tensor((3, 1, 1, 3)))
    b_expanded = SpatialDimension(*[x.expand(1, 2, 3) for x in b.zyx])

    c_matrix = torch.eye(3).reshape(1, 1, 3, 3)
    c_expanded = Rotation.from_matrix(c_matrix.expand(5, 2, 3, 3))

    test = Dummy(a_expanded, b_expanded, c_expanded)

    torch.testing.assert_close(test.a, a)
    torch.testing.assert_close(test.b.z, b.z)
    torch.testing.assert_close(test.b.y, b.y)
    torch.testing.assert_close(test.b.x, b.x)
    torch.testing.assert_close(test.c.as_matrix(), c_matrix)

    assert test.a[0, 0, 0, 0] == 1.0, 'subclass post_init not called'
