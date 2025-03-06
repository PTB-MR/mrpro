"""Tests for the Dictionary Matching Operator."""

import pytest
import torch
from mrpro.operators import DictionaryMatchOp
from mrpro.operators.models import InversionRecovery

from tests import RandomGenerator


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('shape', [(5, 4, 3)])
@pytest.mark.parametrize(
    'index_of_scaling_parameter',
    [None, -2, 0],
    ids=['dont_predict_scale', 'predict_scale_negative_index', 'predict_scale'],
)
def test_dictionaryop_matching(shape: tuple[int], dtype: torch.dtype, index_of_scaling_parameter: int | None) -> None:
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


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64], ids=['float32', 'complex64'])
@pytest.mark.parametrize('shape', [(5, 4, 3)])
@pytest.mark.parametrize(
    'index_of_scaling_parameter',
    [None, -2, 0],
    ids=['dont_predict_scale', 'predict_scale_negative_index', 'predict_scale'],
)
def test_dictionaryop_append(shape: tuple[int], dtype: torch.dtype, index_of_scaling_parameter: int | None) -> None:
    """Test dictionary matching with concatenated entries."""
    rng = RandomGenerator(2)
    m0_1 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1_1 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)

    m0_2 = rng.rand_tensor(shape, dtype=dtype, low=0.2, high=1.0)
    t1_2 = rng.rand_tensor(shape, dtype=dtype.to_real(), low=0.1, high=1.0)

    model = InversionRecovery(rng.float32_tensor(5))

    # dictionary matching when appending the individual tensors
    # concatenation of the tensors
    m0_cat = torch.cat((m0_1, m0_2))
    t1_cat = torch.cat((t1_1, t1_2))
    operator = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    operator.append(m0_1, t1_1)
    operator.append(m0_2, t1_2)
    (y,) = model(m0_cat, t1_cat)
    m0_matched, t1_matched = operator(y)

    # dictionary matching when appending the concatenated tensors
    operator_cat = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    operator_cat.append(m0_cat, t1_cat)
    (y_cat,) = model(m0_cat, t1_cat)
    m0_matched_cat, t1_matched_cat = operator(y_cat)

    if index_of_scaling_parameter is not None:
        torch.testing.assert_close(m0_matched, m0_matched_cat, atol=1e-3, rtol=0.0)
    torch.testing.assert_close(t1_matched, t1_matched_cat, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(
    'index_of_scaling_parameter',
    [None, -2, 0],
    ids=['dont_predict_scale', 'predict_scale_negative_index', 'predict_scale'],
)
def test_dictionaryop_no_entry(index_of_scaling_parameter: int | None) -> None:
    """Test dictionary matching when no entries have been appended before."""
    rng = RandomGenerator(2)
    model = InversionRecovery(rng.float32_tensor(5))

    operator = DictionaryMatchOp(model, index_of_scaling_parameter=index_of_scaling_parameter)
    # create empty signal model
    y = torch.zeros(5, 5, 4, 3)
    with pytest.raises(KeyError, match='No keys in the dictionary. Please first add some x values using `append`.'):
        _ = operator(y)
