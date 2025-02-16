"""Tests for the Dictionary Matching Operator."""

import pytest
import torch
from mrpro.operators import DictionaryMatchOp
from mrpro.operators.models import InversionRecovery
from tests.operators.models.conftest import create_parameter_tensor_tuples
from mrpro.operators import DictionaryMatchOp




def test_real_matching():
    """Test dictionary matching with real-valued signals."""
    model = InversionRecovery(20)
    m0, t1 = create_parameter_tensor_tuples()
    (y,) = model(m0, t1)
    dict_match_op = DictionaryMatchOp(model)
    dictionary = dict_match_op.append(m0, y)
    t1_matched = dict_match_op.forward(y)[1]
    t1_matched = t1_matched.reshape(y.shape[1:])


    (y_matched,) = model(m0, t1_matched)

    assert (y_matched-y).abs().square().mean() < 1e-5


def test_complex_matching():
    """Test dictionary matching with complex-valued signals."""
    model = InversionRecovery(20)
    m0, t1 = create_parameter_tensor_tuples()

    (y,) = model(m0, t1)
    y = torch.complex(y, torch.zeros_like(y))
    dict_match_op = DictionaryMatchOp(model)
    dictionary = dict_match_op.append(m0, y)
    t1_matched = dict_match_op.forward(y)[1]
    t1_matched = t1_matched.reshape(y.shape[1:])


    (y_matched,) = model(m0, t1_matched)

    assert (y_matched-y).abs().square().mean() < 1e-5
