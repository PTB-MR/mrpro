"""Tests for the Transient Inversion Recovery model."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
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
from mrpro.operators.models import TransientInversionRecovery
from tests import RandomGenerator


def create_data(other=10, coils=5, z=100, y=100, x=100):
    random_generator = RandomGenerator(seed=0)
    m0 = random_generator.complex64_tensor(size=(other, coils, z, y, x))
    t1 = random_generator.float32_tensor(size=(other, coils, z, y, x), low=1e-10)
    alpha = random_generator.float32_tensor(size=(other, coils, z, y, x), low=0.01, high=0.1)
    return m0, t1, alpha


@pytest.mark.parametrize(
    ('t', 'result'),
    [
        # short ti
        (0, '-m0'),
        # long ti
        (20, 'm0_star'),
    ],
)
def test_signal_boundaries_single_inversion(t, result):
    """Test for transient inversion recovery.

    Assume single inversion at t=0. Checking that idata output tensor at
    t=0 is close to -m0 and that idata output tensor at large t is close
    to steady state magnetisation m0_star.
    """
    # Repetition time
    tr = 0.005

    model = TransientInversionRecovery(
        signal_time_points=torch.tensor([t]),
        tr=tr,
        inversion_time_points=0,
        delay_inversion_adc=0,
    )
    m0, t1, alpha = create_data()
    (signal,) = model.forward(m0, t1, alpha)

    if result == '-m0':
        torch.testing.assert_close(signal[0, ...], -m0)
    elif result == 'm0_star':
        t1_star = torch.div(1, torch.div(1, t1) - torch.div(torch.log(torch.cos(alpha)), tr))
        m0_star = torch.div(m0 * t1_star, t1)
        torch.testing.assert_close(signal[0, ...], m0_star)


@pytest.mark.parametrize(
    ('parameter_name'),
    ['tr', 'inversion_time_points', 'delay_inversion_adc', 'first_adc_time_point'],
)
def test_invalid_shapes_of_input_parameter(parameter_name):
    """Ensure error message for invalid shapes."""
    random_generator = RandomGenerator(seed=0)
    parameter_dict = {
        'signal_time_points': random_generator.float32_tensor((5, 4)),
        'tr': 5,
        'inversion_time_points': 0,
        'delay_inversion_adc': 0.02,
        'first_adc_time_point': 0,
    }
    parameter_dict[parameter_name] = random_generator.float32_tensor(6)
    with pytest.raises(ValueError, match=f'Broadcasted shape of {parameter_name} does not match'):
        TransientInversionRecovery(**parameter_dict)


def test_invalid_signal_during_tau():
    """Ensure error message for t before data acquisition."""
    with pytest.raises(ValueError, match='No data points should lie between inversion'):
        TransientInversionRecovery(
            signal_time_points=torch.tensor([0]),
            tr=0.005,
            inversion_time_points=0,
            delay_inversion_adc=0.02,
        )


def test_invalid_signal_before_inv():
    """Ensure error message for t before inversion."""
    with pytest.raises(ValueError, match='If data has been acquired before the first'):
        TransientInversionRecovery(
            signal_time_points=torch.tensor([-1]),
            tr=0.005,
            inversion_time_points=0,
            delay_inversion_adc=0.02,
        )

    with pytest.raises(ValueError, match='Acquisitions detected before start of acquisition'):
        TransientInversionRecovery(
            signal_time_points=torch.tensor([-1]),
            tr=0.005,
            inversion_time_points=torch.tensor([0]),
            delay_inversion_adc=0.02,
            first_adc_time_point=-0.5,
        )
