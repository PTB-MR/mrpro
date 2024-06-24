"""Tests for transient steady state signal model."""

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
from mrpro.operators.models import TransientSteadyStateWithPreparation
from tests.conftest import SHAPE_VARIATIONS_SIGNAL_MODELS
from tests.conftest import create_parameter_tensor_tuples


@pytest.mark.parametrize(
    ('sampling_time', 'm0_scaling_preparation', 'result'),
    [
        (0, 1, 'm0'),  # short sampling time without preparation pulse
        (0, 0, '0'),  # short sampling time after saturation pulse
        (0, -1, '-m0'),  # short sampling time after inversion pulse
        (20, 1, 'm0*'),  # long sampling time without preparation pulse
        (20, 0, 'm0*'),  # long sampling time after saturation pulse
        (20, -1, 'm0*'),  # long sampling time after inversion pulse
    ],
)
def test_transient_steady_state(sampling_time, m0_scaling_preparation, result):
    """Test transient steady state for very long and very short times."""
    repetition_time = 5
    model = TransientSteadyStateWithPreparation(sampling_time, repetition_time, m0_scaling_preparation)
    m0, t1, flip_angle = create_parameter_tensor_tuples(number_of_tensors=3)
    (signal,) = model.forward(m0, t1, flip_angle)

    # Assert closeness to m0
    if result == 'm0':
        torch.testing.assert_close(signal[0, ...], m0)
    # Assert closeness to 0
    elif result == '0':
        torch.testing.assert_close(signal[0, ...], torch.zeros_like(m0))
    # Assert closeness to -m0
    elif result == '-m0':
        torch.testing.assert_close(signal[0, ...], -m0)
    # Assert closensess to m0*
    elif result == 'm0*':
        t1 = torch.where(t1 == 0, 1e-10, t1)
        t1_star = 1 / (1 / t1 - torch.log(torch.cos(flip_angle)) / repetition_time)
        m0_star = m0 * t1_star / t1
        torch.testing.assert_close(signal[0, ...], m0_star)


def test_transient_steady_state_inversion_recovery():
    """Transient steady state as inversion recovery.

    For very small flip angles and long repetition times, the transient steady state should be the same as a
    inversion recovery model.
    """
    t1 = torch.as_tensor([100, 200, 300, 400, 500, 1000, 2000, 4000])
    flip_angle = torch.ones_like(t1) * 0.0001
    m0 = torch.ones_like(t1)
    sampling_time = torch.as_tensor([0, 100, 400, 800, 2000])

    # analytical signal
    analytical_signal = m0 * (1 - 2 * torch.exp(-(sampling_time[..., None] / t1)))

    model = TransientSteadyStateWithPreparation(sampling_time, repetition_time=100, m0_scaling_preparation=-1)
    (signal,) = model.forward(m0, t1, flip_angle)

    torch.testing.assert_close(signal, analytical_signal)


@SHAPE_VARIATIONS_SIGNAL_MODELS
def test_transient_steady_state_shape(parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    (sampling_time,) = create_parameter_tensor_tuples(contrast_dim_shape, number_of_tensors=1)
    model_op = TransientSteadyStateWithPreparation(sampling_time, repetition_time=5)
    m0, t1, flip_angle = create_parameter_tensor_tuples(parameter_shape, number_of_tensors=3)
    (signal,) = model_op.forward(m0, t1, flip_angle)
    assert signal.shape == signal_shape
