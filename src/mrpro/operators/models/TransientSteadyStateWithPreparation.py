"""Transient steady state signal models."""

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

import torch

from mrpro.operators.SignalModel import SignalModel


class TransientSteadyStateWithPreparation(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for transient steady state.

    This signal model describes the behavior of the longitudinal magnetisation during continuous acquisition after
    a preparation pulse. The effect of the preparation pulse is modelled by a scaling factor applied to the
    equilibrium magnetisation. T2 effects are not considered here.

    Mz(t) = M0* + (s*M0 - M0*)e^(-t / T1*)

    with
    M0 equilibrium magnetisation
    s scaling factor describing preparation pulse (e.g. s = -1 for inversion pulse)
    Mz(t) measured magnetisation at time point t
    M0* steady-state magnetisation
    T1* effective longitudinal magnetisation

    """

    def __init__(
        self,
        sampling_time: float | torch.Tensor,
        repetition_time: float,
        m0_scaling_preparation: float = 1.0,
        delay_after_preparation: float = 0.0,
    ):
        """Initialize transient steady state signal model.

        Parameters
        ----------
        sampling_time
            time points when model is evaluated
            with shape (time, ...)
        repetition_time
            repetition time
        m0_scaling_preparation
            scaling of the equilibrium magnetisation due to the preparation pulse before the data acquisition
        delay_after_preparation
            Time between preparation pulse and start of data acquisition.
            During this time standard longitudinal relaxation occurs.

        """
        super().__init__()
        sampling_time = torch.as_tensor(sampling_time)
        self.sampling_time = torch.nn.Parameter(sampling_time, requires_grad=sampling_time.requires_grad)
        self.repetition_time = repetition_time
        self.m0_scaling_preparation = m0_scaling_preparation
        self.delay_after_preparation = delay_after_preparation

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, flip_angle: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply transient steady state signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1
            with shape (... other, coils, z, y, x)
        flip_angle
            flip angle of data acquisition
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape (time ... other, coils, z, y, x)
        """
        delta_ndim = m0.ndim - (self.sampling_time.ndim - 1)  # -1 for time
        sampling_time = self.sampling_time[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.sampling_time

        # effect of preparation pulse
        m_start = m0 * self.m0_scaling_preparation

        # relaxation towards M0
        t1 = torch.where(t1 == 0, 1e-10, t1)
        m_start = m0 + (m_start - m0) * torch.exp(-(self.delay_after_preparation / t1))

        # transient steady state
        t1_star = 1 / (1 / t1 - torch.log(torch.cos(flip_angle)) / self.repetition_time)
        m0_star = m0 * t1_star / t1
        signal = m0_star + (m_start - m0_star) * torch.exp(-sampling_time / t1_star)
        return (signal,)
