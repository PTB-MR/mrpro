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
    equilibrium magnetisation. A delay after the preparation pulse can be defined. During this time T1 relaxation to M0
    occurs. Data acquisition starts after this delay. Perfect spoiling is assumed and hence T2 effects are not
    considered in the model. In addition this model assumes TR << T1 and TR << T1* (see definition below). More
    information can be found here:

    Deichmann, R. & Haase, A. Quantification of T1 values by SNAPSHOT-FLASH NMR imaging. J. Magn. Reson. 612, 608-612
    (1992) [http://doi.org/10.1016/0022-2364(92)90347-A].
    Look, D. C. & Locker, D. R. Time Saving in Measurement of NMR and EPR Relaxation Times. Rev. Sci. Instrum 41, 250
    (1970) [https://doi.org/10.1063/1.1684482].

    Let's assume we want to describe a continuous acquisition after an inversion pulse, then we have three parts:
    [Part A: 180° inversion pulse][Part B: spoiler gradient][Part C: Continuous data acquisition]

    Part A: The 180° pulse leads to an inversion of the equilibrium magnetisation (M0) to -M0. This can be described by
            setting the scaling factor m0_scaling_preparation to -1

    Part B: Commonly after an inversion pulse a strong spoiler gradient is played out to compensate for non-perfect
            inversion. During this time the magnetisation follows Mz(t) the signal model:
                    Mz(t) = M0 + (m0_scaling_preparation*M0 - M0)e^(-t / T1)

    Part C: After the spoiler gradient the data acquisition starts and the magnetisation Mz(t) can be described by the
            signal model:
                    Mz(t) = M0* + (M0_init - M0*)e^(-t / T1*)
            where the initial magnetisation is
                    M0_init = M0 + (m0_scaling_preparation*M0 - M0)e^(-delay_after_preparation / T1)
            the effective longitudinal relaxation time is
                    T1* = 1/(1/T1 - 1/repetition_time ln(cos(flip_angle)))
            and the steady-state magnetisation is
                    M0* = M0 T1* / T1

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
            time points when model is evaluated. A sampling_time of 0 describes the first acquired data point after the
            inversion pulse and spoiler gradients. To take the T1 relaxation during the delay between inversion pulse
            and start of data acquisition into account, set the delay_after_preparation > 0.
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
            flip angle of data acquisition in rad
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
