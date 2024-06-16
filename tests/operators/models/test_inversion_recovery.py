"""Tests for inversion recovery signal model."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import torch
from mrpro.operators.models import InversionRecovery
from tests.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS
from tests.conftest import create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '-m0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_inversion_recovery(ti, result):
    """Test for inversion recovery.

    Checking that idata output tensor at ti=0 is close to -m0. Checking
    that idata output tensor at large ti is close to m0.
    """
    model = InversionRecovery(ti)
    m0, t1 = create_parameter_tensor_tuples()
    (image,) = model.forward(m0, t1)

    # Assert closeness to -m0 for ti=0
    if result == '-m0':
        torch.testing.assert_close(image[0, ...], -m0)
    # Assert closeness to m0 for large ti
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_inversion_recovery_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (ti,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = InversionRecovery(ti)
    m0, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=2)
    (signal,) = model_op.forward(m0, t1)
    assert signal.shape == signal_shape
