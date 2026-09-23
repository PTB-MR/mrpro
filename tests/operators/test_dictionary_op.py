"""Test the dictionary operator."""

import pytest
from mrpro.operators import DictionaryOp
from mrpro.utils import RandomGenerator

from tests.helper import dotproduct_adjointness_test


@pytest.mark.parametrize(
    'dimensions',
    [
        (-5, -1, -2),
        (0, 1),
        (0, -5, 1),
    ],
)
def test_dictionaryop_adjointness(dimensions: tuple[int, ...]) -> None:
    """Test the adjointness of the dictionary operator."""

    rng = RandomGenerator(seed=0)

    patch_size = (5, 15, 10, 2, 2, 2)
    dictionary = rng.complex64_tensor(size=(256, 8))

    u = rng.complex64_tensor(size=patch_size)
    v = rng.complex64_tensor(size=(5, 15, 10, 256))

    op = DictionaryOp(
        dictionary=dictionary,
        dim=dimensions,
        patch_size=patch_size,
    )

    dotproduct_adjointness_test(op, u, v)
