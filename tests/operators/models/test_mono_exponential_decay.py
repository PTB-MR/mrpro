"""Tests for the mono-exponential decay signal model."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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
from mrpro.operators.models import MonoExponentialDecay
from tests.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS
from tests.conftest import create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('decay_time', 'result'),
    [
        (0, 'm0'),  # short decay time
        (20, '0'),  # long decay time
    ],
)
def test_mono_exponential_decay(decay_time, result):
    """Test for mono-exponential decay signal model.

    Checking that idata output tensor at ti=0 is close to 0. Checking
    that idata output tensor at large ti is close to m0.
    """
    model = MonoExponentialDecay(decay_time)
    m0, decay_constant = create_parameter_tensor_tuples()
    (image,) = model.forward(m0, decay_constant)

    zeros = torch.zeros_like(m0)

    # Assert closeness to m0 for short decay_time
    if result == '0':
        torch.testing.assert_close(image[0, ...], zeros)
    # Assert closeness to 0 for large decay_time
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_mono_exponential_decay_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (decay_time,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = MonoExponentialDecay(decay_time)
    m0, decay_constant = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=2)
    (signal,) = model_op.forward(m0, decay_constant)
    assert signal.shape == signal_shape
