"""Tests for the Dictionary Matching Operator."""

import pytest
import torch
from mrpro.operators import DictionaryMatchOp
from mrpro.operators.models import InversionRecovery

from tests import RandomGenerator


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('shape', [(5, 4, 3)])
@pytest.mark.parametrize('index_of_scaling_parameter', [None, 0], ids=['dont_predict_scale', 'predict_scale'])
def test_dictionaryop_matching(shape: tuple[int], dtype: torch.dtype, index_of_scaling_parameter: int | None):
    """Test dictionary matching with real-valued signals."""
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))
    m0 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)
    (y,) = model(m0, t1)

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    operator.append(m0, t1)

    m0_matched, t1_matched = operator(y)

    if index_of_scaling_parameter is not None:
        torch.testing.assert_close(m0_matched, m0, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(t1_matched, t1, atol=1e-4, rtol=0.0)


# TODO:
# doing two appends should result in the same .y and .x as doing one append with the concatenated inputs
# empty dictionary should print error message (google: with pytest.raises(KeyError)
# Add Error message if the scaling arguments is out of range
# spell and grammar check
