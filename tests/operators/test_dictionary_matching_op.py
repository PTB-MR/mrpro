"""Tests for the Dictionary Matching Operator."""

import pytest
import torch
from mrpro.operators import DictionaryMatchOp
from mrpro.operators.models import InversionRecovery

from tests import RandomGenerator


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('shape', [(5, 4, 3)])
@pytest.mark.parametrize('scaling_argument', [None, 0], ids=['dont_predict_scale', 'predict_scale'])
def test_dictionaryop_matching(shape: tuple[int], dtype: torch.dtype, scaling_argument: int | None):
    """Test dictionary matching with real-valued signals."""
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))
    m0 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)
    (y,) = model(m0, t1)

    operator = DictionaryMatchOp(model, scaling_argument=scaling_argument)
    operator.append(m0[:1], t1[:1])
    operator.append(m0[1:], t1[1:])

    m0_matched, t1_matched = operator(y)

    if scaling_argument is not None:
        torch.testing.assert_close(m0_matched, m0, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(t1_matched, t1, atol=1e-4, rtol=0.0)
