"""Tests for MOLLI signal model."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from mrpro.operators.models import MOLLI
from tests.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS
from tests.conftest import create_parameter_tensor_tuples
from tests.helper import autodiff_of_operator_test


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, 'a-b'),  # short ti
        (20, 'a'),  # long ti
    ],
)
def test_molli(ti, result):
    """Test for MOLLI.

    Checking that idata output tensor at ti=0 is close to a. Checking
    that idata output tensor at large ti is close to a-b.
    """
    # Generate qdata tensor, not random as a<b is necessary for t1_star to be >= 0
    other, coils, z, y, x = 10, 5, 100, 100, 100
    a = torch.ones((other, coils, z, y, x)) * 2
    b = torch.ones((other, coils, z, y, x)) * 5
    t1 = torch.ones((other, coils, z, y, x)) * 2

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model.forward(a, b, t1)

    # Assert closeness to a-b for large ti
    if result == 'a-b':
        torch.testing.assert_close(image[0, ...], a - b)
    # Assert closeness to a for ti=0
    elif result == 'a':
        torch.testing.assert_close(image[0, ...], a)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_molli_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (ti,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = MOLLI(ti)
    a, b, t1 = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=3)
    (signal,) = model_op.forward(a, b, t1)
    assert signal.shape == signal_shape


@pytest.mark.filterwarnings('ignore:Anomaly Detection has been enabled')
def test_autodiff_molli():
    """Test autodiff works for molli model."""
    model = MOLLI(ti=10)
    a, b, t1 = create_parameter_tensor_tuples((2, 5, 10, 10, 10), number_of_tensors=3)
    autodiff_of_operator_test(model, a, b, t1)
