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

from mrpro.operators._SignalModel import SignalModel


class EpgRfPulse:
    """Mixing of EPG configuration states due to RF pulse."""

    def __init__(self, flip_angle: torch.Tensor, phase: torch.Tensor, b1_scaling_factor: torch.Tensor | None = None):
        """Initialise the rotation matrix describing the RF pulse.

        Parameters
        ----------
        flip_angle
            Flip angle of the RF pulse in rad
        phase
            Phase of the RF pulse
        b1_scaling_factor
            Scaling of flip angle due to B1 inhomogeneities
        """
        if b1_scaling_factor is not None:
            flip_angle = flip_angle * b1_scaling_factor[None, ...]
        cosa = torch.cos(flip_angle)
        sina = torch.sin(flip_angle)
        cosa2 = (cosa + 1) / 2
        sina2 = 1 - cosa2

        ejp = torch.exp(1j * phase)
        inv_ejp = 1 / ejp

        self.rf_rotation_matrix: torch.Tensor = torch.stack(
            [
                cosa2 + 0j,
                ejp**2 * sina2,
                -1j * ejp * sina,
                inv_ejp**2 * sina2,
                cosa2 + 0j,
                1j * inv_ejp * sina,
                -1j / 2.0 * inv_ejp * sina,
                1j / 2.0 * ejp * sina,
                cosa + 0j,
            ],
            -1,
        ).reshape(*flip_angle.shape, 3, 3)

    def apply(self, epg_configuration_states: torch.Tensor) -> torch.Tensor:
        """Propagate EPG states through an RF rotation.

        Parameters
        ----------
        epg_configuration_states
            EPG configuration states Fplus, Fminus, Z

        Returns
        -------
            EPG configuration states after RF pulse
        """
        return torch.matmul(self.rf_rotation_matrix, epg_configuration_states)


class EpgGradient:
    """Dephasing and Rephasing due to gradient."""

    def __init__(self, keep_fixed_number_of_states: bool = False):
        """Gradient de- and rephasing.

        Parameters
        ----------
        keep_fixed_number_of_states
            True to NOT add any higher-order states - assume that they just go to zero.  Be careful - this speeds up
            simulations, but may compromise accuracy!
        """
        self.keep_fixed_number_of_states: bool = keep_fixed_number_of_states

    def apply(self, epg_configuration_states: torch.Tensor) -> torch.Tensor:
        """Propagate EPG states through a gradient.

        Parameters
        ----------
        epg_configuration_states
            EPG configuration states Fplus, Fminus, Z

        Returns
        -------
            EPG configuration states after gradient
        """
        zero = torch.zeros(
            *epg_configuration_states.shape[:-2],
            1,
            device=epg_configuration_states.device,
            dtype=epg_configuration_states.dtype,
        )
        if self.keep_fixed_number_of_states:
            f_plus = torch.cat(
                (
                    epg_configuration_states[..., 1, 1:2].conj() if epg_configuration_states.shape[-1] > 1 else zero,
                    epg_configuration_states[..., 0, :-1],
                ),
                -1,
            )
            f_minus = torch.cat((epg_configuration_states[..., 1, 1:], zero), -1)
            z = epg_configuration_states[..., 2, :]
        else:
            f_plus = torch.cat(
                (
                    epg_configuration_states[..., 1, 1:2].conj() if epg_configuration_states.shape[-1] > 1 else zero,
                    epg_configuration_states[..., 0, :],
                ),
                -1,
            )
            f_minus = torch.cat((epg_configuration_states[..., 1, 1:], zero, zero), -1)
            z = torch.cat((epg_configuration_states[..., 2, :], zero), -1)
        return torch.stack((f_plus, f_minus, z), -2)


class EpgRelaxation:
    def __init__(self, relaxation_time: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t1_recovery: bool = True):
        exp_t2 = torch.exp(-relaxation_time / t2)
        exp_t1 = torch.exp(-relaxation_time / t1)
        exp_t1, exp_t2 = torch.broadcast_tensors(exp_t1, exp_t2)
        self.relaxation_matrix: torch.Tensor = torch.stack([exp_t2, exp_t2, exp_t1], dim=-1)
        self.t1_recovery: bool = t1_recovery

    def apply(self, epg_configuration_states: torch.Tensor) -> torch.Tensor:
        """Propagate EPG states through a period of relaxation and recovery.

        Parameters
        ----------
        epg_configuration_states
            EPG configuration states Fplus, Fminus, Z

        Returns
        -------
            EPG configuration states after relaxation and recovery
        """
        epg_configuration_states = self.relaxation_matrix[..., None] * epg_configuration_states

        if self.t1_recovery:
            epg_configuration_states[..., 2, 0] = epg_configuration_states[..., 2, 0] + (
                1 - self.relaxation_matrix[..., -1]
            )
        return epg_configuration_states


class EpgMrfFisp(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for classic Magnetic resonance fingerprinting (MRF).

    This signal model describes a classic MRF scan with a spoiled gradient echo acquisition. The sequence starts with an
    inversion pulse to make the sequence more sensitive to T1 and is then followed by a train of RF pulses. The flip
    angle and phase of the RF pulse and the echo time and repetition time can be varied for each acquisition point.

    """

    def __init__(
        self,
        flip_angles: float | torch.Tensor,
        rf_phases: float | torch.Tensor,
        te: float | torch.Tensor,
        tr: float | torch.Tensor,
        ti_value: float = 0.0,
        max_n_configuration_states: int | float = torch.inf,
    ):
        """Initialize MRF signal model.

        Parameters
        ----------
        flip_angles
            flip angles of excitation RF pulses in rad
            with shape (time, ...)
        rf_phases
            phase of excitation RF pulses in rad
            with shape (time, ...)
        te
            echo times
            with shape (time, ...)
        tr
            repetition times
            with shape (time, ...)
        ti_value
            inversion time between inversion pulse at the beginning of the sequence and first acquisition
        max_n_configuration_states
            maximum number of configuration states to be considered. Default means all configuration states are
            considered.
        """
        super().__init__()

        try:
            flip_angles, rf_phases, te, tr = torch.broadcast_tensors(flip_angles, rf_phases, te, tr)
        except RuntimeError:
            # Not broadcastable
            raise ValueError(
                'Shapes of flip_angles, rf_phases, te and tr do not match and also cannot be broadcasted.',
            ) from None

        # convert all parameters to tensors
        flip_angles = torch.as_tensor(flip_angles)
        rf_phases = torch.as_tensor(rf_phases)
        te = torch.as_tensor(te)
        tr = torch.as_tensor(tr)
        ti = torch.as_tensor(ti_value)

        # nn.Parameters allow for grad calculation
        self.flip_angles = torch.nn.Parameter(flip_angles, requires_grad=flip_angles.requires_grad)
        self.rf_phases = torch.nn.Parameter(rf_phases, requires_grad=rf_phases.requires_grad)
        self.te = torch.nn.Parameter(te, requires_grad=te.requires_grad)
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

        self.max_n_configuration_states = max_n_configuration_states

    def forward(self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply MRF signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        t1
            t1 times
            with shape (... other, coils, z, y, x)
        t2
            t2 times
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal of a MRF acquisition
            with shape (time ... other, coils, z, y, x)
        """
        delta_ndim = m0.ndim - (self.flip_angles.ndim - 1)  # -1 for time
        flip_angles = self.flip_angles[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.flip_angles
        rf_phases = self.rf_phases[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.rf_phases
        te = self.te[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.te
        tr = self.tr[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.tr

        signal = torch.zeros(flip_angles.shape[0], *m0.shape, dtype=torch.cfloat, device=m0.device)
        epg_configuration_states = torch.zeros((*m0.shape, 3, 1), dtype=torch.cfloat, device=m0.device)

        # Inversion pulse as preparation
        epg_configuration_states[..., :, 0] = torch.tensor((0.0, 0, -1.0))
        # Relaxation after inversion pulse
        epg_configuration_states = EpgRelaxation(self.ti, t1, t2).apply(epg_configuration_states)

        # RF-pulse -> relaxation during TE -> get signal -> gradient -> relaxation during TR-TE
        for i in range(flip_angles.shape[0]):
            rf_pulse = EpgRfPulse(flip_angles[i, ...], rf_phases[i, ...])
            te_relaxation = EpgRelaxation(te[i, ...], t1, t2)
            tr_relaxation = EpgRelaxation((tr - te)[i, ...], t1, t2)
            if i == 0 or epg_configuration_states.shape[0] >= self.max_n_configuration_states:
                gradient_dephasing = EpgGradient(epg_configuration_states.shape[-1] >= self.max_n_configuration_states)

            epg_configuration_states = rf_pulse.apply(epg_configuration_states)
            epg_configuration_states = te_relaxation.apply(epg_configuration_states)
            signal[i, ...] = m0 * epg_configuration_states[..., 0, 0]
            epg_configuration_states = gradient_dephasing.apply(epg_configuration_states)
            epg_configuration_states = tr_relaxation.apply(epg_configuration_states)
        return (signal,)


class EpgTse(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for turbo spin echo (TSE) with multiple echo trains.

    This signal model describes a turbo spin sequence. The sequence starts with a 90° excitation pulse with a phase of
    90°. Afterwards a train of refocusing pulses follow. If more than one repetition time is specified, this is repeated
    again for each repetition time. Relaxation takes place in between the different echo trains.
    """

    def __init__(
        self,
        flip_angles: float | torch.Tensor,
        rf_phases: float | torch.Tensor,
        te: float | torch.Tensor,
        tr: float | torch.Tensor = 0.0,
        max_n_configuration_states: int | float = torch.inf,
    ):
        """Initialize TSE signal model.

        Parameters
        ----------
        flip_angles
            flip angles of refocusing pulses in rad
            with shape (echoes, ...)
        rf_phases
            phase of refocusing pulses in rad
            with shape (echoes, ...)
        te
            echo times of a spin echo train
            with shape (echoes, ...)
        tr
            repetition time between echo trains. Default corresponds to single echo train
            with shape (number of echo trains,)
        max_n_configuration_states
            maximum number of configuration states to be considered. Default means all configuration states are
            considered.
        """
        super().__init__()

        try:
            (flip_angles, rf_phases, te) = torch.broadcast_tensors(flip_angles, rf_phases, te)
        except RuntimeError:
            # Not broadcastable
            raise ValueError(
                'Shapes of flip_angles, rf_phases and te do not match and also cannot be broadcasted.',
            ) from None

        # convert all parameters to tensors
        flip_angles = torch.as_tensor(flip_angles)
        rf_phases = torch.as_tensor(rf_phases)
        te = torch.as_tensor(te)
        tr = torch.as_tensor(torch.atleast_1d(torch.as_tensor(tr)))

        # nn.Parameters allow for grad calculation
        self.flip_angles = torch.nn.Parameter(flip_angles, requires_grad=flip_angles.requires_grad)
        self.rf_phases = torch.nn.Parameter(rf_phases, requires_grad=rf_phases.requires_grad)
        self.te = torch.nn.Parameter(te, requires_grad=te.requires_grad)
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)

        self.max_n_configuration_states = max_n_configuration_states

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, b1_scaling_factor: torch.Tensor
    ) -> tuple[torch.Tensor,]:
        """Apply TSE signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        t1
            t1 times
            with shape (... other, coils, z, y, x)
        t2
            t2 times
            with shape (... other, coils, z, y, x)
        b1_scaling_factor
            scaling factor of flip angles of refocusing pulses
            with shape (... other, coils, z, y, x)

        Returns
        -------
            Signal of a TSE acquisition
        """
        delta_ndim = m0.ndim - (self.flip_angles.ndim - 1)  # -1 for time
        flip_angles = self.flip_angles[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.flip_angles
        rf_phases = self.rf_phases[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.rf_phases
        te = self.te[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.te

        signal = torch.zeros(flip_angles.shape[0] * len(self.tr), *m0.shape, dtype=torch.cfloat, device=m0.device)
        epg_configuration_states = torch.zeros((*m0.shape, 3, 1), dtype=torch.cfloat, device=m0.device)

        # Define 90° excitation pulse
        rf_excitation_pulse = EpgRfPulse(
            torch.as_tensor([torch.pi / 2], device=flip_angles.device),
            torch.as_tensor([torch.pi / 2], device=flip_angles.device),
        )

        epg_configuration_states[..., :, 0] = torch.tensor((0.0, 0, 1.0))
        # Go through echo trains separated by TR
        # 90° excitation pulse -> echo train (see below) -> relaxation during TR - (sum over all TEs)
        for j in range(len(self.tr)):
            if j == 0 or self.tr[j] != self.tr[j - 1]:
                tr_relaxation = EpgRelaxation((self.tr[j] - torch.sum(te)), t1, t2)
            epg_configuration_states = rf_excitation_pulse.apply(epg_configuration_states)
            # Go through refocusing pulses (i.e one echo train)
            # relaxation during TE/2 -> gradient -> refocusing pulse -> gradient -> relaxation during TE/2 -> get signal
            for i in range(flip_angles.shape[0]):
                rf_refocusing_pulse = EpgRfPulse(flip_angles[i, ...], rf_phases[i, ...], b1_scaling_factor)
                te_half_relaxation = EpgRelaxation(te[i, ...] / 2, t1, t2)
                if i == 0 or epg_configuration_states.shape[0] >= self.max_n_configuration_states:
                    gradient_dephasing = EpgGradient(
                        epg_configuration_states.shape[-1] >= self.max_n_configuration_states
                    )

                epg_configuration_states = te_half_relaxation.apply(epg_configuration_states)
                epg_configuration_states = gradient_dephasing.apply(epg_configuration_states)
                epg_configuration_states = rf_refocusing_pulse.apply(epg_configuration_states)
                epg_configuration_states = gradient_dephasing.apply(epg_configuration_states)
                epg_configuration_states = te_half_relaxation.apply(epg_configuration_states)
                signal[i + j * flip_angles.shape[0], ...] = m0 * epg_configuration_states[..., 0, 0]

            epg_configuration_states = tr_relaxation.apply(epg_configuration_states)

        return (signal,)


# Parts of this code were adapted from https://github.com/fzimmermann89/epgtorch/tree/master
#
# Copyright (c) 2022 Felix
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
