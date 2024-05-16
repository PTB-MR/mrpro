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
        flip_angle, phase = flip_angle.moveaxis(-1, 0), phase.moveaxis(-1, 0)
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

    def forward(self, epg_configuration_states: torch.Tensor) -> torch.Tensor:
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

    def forward(self, epg_configuration_states: torch.Tensor) -> torch.Tensor:
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

    def forward(self, epg_configuration_states: torch.Tensor) -> torch.Tensor:
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


class EpgMrfFisp(SignalModel[torch.Tensor, torch.Tensor]):
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
        ti: float | torch.Tensor,
        max_n_configuration_states: int,
    ):
        """Initialize MRF signal model.

        Parameters
        ----------
        flip_angles
            flip angles of excitation RF pulses in rad
        rf_phases
            phase of excitation RF pulses in rad
        te
            echo times
        tr
            repetition times
        ti
            inversion time between inversion pulse at the beginning of the sequence and first acquisition
        max_n_configuration_states
            maximum number of configuration states to be considered
        """
        super().__init__()
        flip_angles = torch.as_tensor(flip_angles)
        self.flip_angles = torch.nn.Parameter(flip_angles, requires_grad=flip_angles.requires_grad)
        rf_phases = torch.as_tensor(rf_phases)
        self.rf_phases = torch.nn.Parameter(rf_phases, requires_grad=rf_phases.requires_grad)
        te = torch.as_tensor(te)
        self.te = torch.nn.Parameter(te, requires_grad=te.requires_grad)
        tr = torch.as_tensor(tr)
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)
        ti = torch.as_tensor(ti)
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)
        
        self.max_n_configuration_states: int = max_n_configuration_states

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply MRF signal model.

        Parameters
        ----------
        t1
            t1 times
        t2
            t2 times

        Returns
        -------
            signal of a MRF acquisition
        """
        flip_angles, rf_phases, te, tr, t1, t2 = (
            torch.atleast_1d(i) for i in (self.flip_angles, self.rf_phases, self.te, self.tr, t1, t2)
        )
        flip_angles, rf_phases = torch.broadcast_tensors(flip_angles, rf_phases)
        shape_pulses = torch.broadcast_shapes(flip_angles.shape, rf_phases.shape, te.shape, tr.shape)
        shape_prop = torch.broadcast_shapes(t1.shape, t2.shape)
        shape_common = torch.broadcast_shapes(shape_prop, shape_pulses[:-1])
        batch_sizes = shape_common
        n_rf_pulses = shape_pulses[-1]

        signal = torch.zeros(*batch_sizes, n_rf_pulses, dtype=torch.cfloat, device=flip_angles.device)
        epg_configuration_states = torch.zeros((*batch_sizes, 3, 1), dtype=torch.cfloat, device=flip_angles.device)

        # Inversion pulse as preparation
        epg_configuration_states[..., :, 0] = torch.tensor((0.0, 0, -1.0))
        # Relaxation after inversion pulse
        epg_configuration_states = EpgRelaxation(self.ti, t1, t2).forward(epg_configuration_states)

        # RF-pulse -> relaxation during TE -> get signal -> gradient dephasing -> relaxation during TR-TE
        for i in range(n_rf_pulses):
            if i == 0 or flip_angles.shape[-1] > 1:
                rf_pulse = EpgRfPulse(flip_angles[..., i], rf_phases[..., i])
            if i == 0 or te.shape[-1] > 1:
                te_relaxation = EpgRelaxation(te[..., i], t1, t2)
            if i == 0 or te.shape[-1] > 1 or tr.shape[-1] > 1:
                tr_relaxation = EpgRelaxation((tr - te)[..., i], t1, t2)
            if i == 0 or epg_configuration_states.shape[-1] >= self.max_n_configuration_states:
                gradient_dephasing = EpgGradient(epg_configuration_states.shape[-1] >= self.max_n_configuration_states)

            epg_configuration_states = rf_pulse.forward(epg_configuration_states)
            epg_configuration_states = te_relaxation.forward(epg_configuration_states)
            signal[..., i] = epg_configuration_states[..., 0, 0]
            epg_configuration_states = gradient_dephasing.forward(epg_configuration_states)
            epg_configuration_states = tr_relaxation.forward(epg_configuration_states)
        return (signal,)


class EpgTse(SignalModel[torch.Tensor, torch.Tensor]):
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
        tr: float | torch.Tensor,
        max_n_configuration_states: int,
    ):
        """Initialize TSE signal model.

        Parameters
        ----------
        flip_angles
            flip angles of refocusing pulses in rad
        rf_phases
            phase of refocusing pulses in rad
        te
            echo times of a spin echo train
        tr
            repetition time between echo trains
        max_n_configuration_states
            maximum number of configuration states to be considered
        """
        super().__init__()
        flip_angles = torch.as_tensor(flip_angles)
        self.flip_angles = torch.nn.Parameter(flip_angles, requires_grad=flip_angles.requires_grad)
        rf_phases = torch.as_tensor(rf_phases)
        self.rf_phases = torch.nn.Parameter(rf_phases, requires_grad=rf_phases.requires_grad)
        te = torch.as_tensor(te)
        self.te = torch.nn.Parameter(te, requires_grad=te.requires_grad)
        tr = torch.as_tensor(tr)
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)
        
        self.max_n_configuration_states: int = max_n_configuration_states

    def forward(self, t1: torch.Tensor, t2: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply TSE signal model.

        Parameters
        ----------
        t1
            t1 times
        t2
            t2 times

        Returns
        -------
            Signal of a TSE acquisition
        """
        flip_angles, rf_phases, te, tr, t1, t2 = (
            torch.atleast_1d(i) for i in (self.flip_angles, self.rf_phases, self.te, self.tr, t1, t2)
        )
        flip_angles, rf_phases = torch.broadcast_tensors(flip_angles, rf_phases)
        shape_pulses = torch.broadcast_shapes(flip_angles.shape, rf_phases.shape, te.shape)
        shape_prop = torch.broadcast_shapes(t1.shape, t2.shape)
        shape_common = torch.broadcast_shapes(shape_prop, shape_pulses[:-1])
        batch_sizes = shape_common
        n_refocusing_pulses = shape_pulses[-1]

        signal = torch.zeros(*batch_sizes, n_refocusing_pulses * len(tr), dtype=torch.cfloat, device=flip_angles.device)
        epg_configuration_states = torch.zeros((*batch_sizes, 3, 1), dtype=torch.cfloat, device=flip_angles.device)

        # Define 90° excitation pulse
        rf_excitation_pulse = EpgRfPulse(
            torch.as_tensor([torch.pi / 2], device=flip_angles.device),
            torch.as_tensor([torch.pi / 2], device=flip_angles.device),
        )

        epg_configuration_states[..., :, 0] = torch.tensor((0.0, 0, 1.0))
        # Go through echo trains separated by TR
        for j in range(len(tr)):
            tr_relaxation = EpgRelaxation((tr[j] - torch.sum(te)), t1, t2)
            epg_configuration_states = rf_excitation_pulse.forward(epg_configuration_states)
            # Go through refocusing pulses (i.e one echo train)
            for i in range(n_refocusing_pulses):
                if i == 0 or flip_angles.shape[-1] > 1:
                    rf_refocusing_pulse = EpgRfPulse(flip_angles[..., i], rf_phases[..., i])
                if i == 0 or te.shape[-1] > 1:
                    te_half_relaxation = EpgRelaxation(te[..., i] / 2, t1, t2)
                if i == 0 or epg_configuration_states.shape[-1] >= self.max_n_configuration_states:
                    gradient_dephasing = EpgGradient(epg_configuration_states.shape[-1] >= self.max_n_configuration_states)


                epg_configuration_states = te_half_relaxation.forward(epg_configuration_states)
                epg_configuration_states = gradient_dephasing.forward(epg_configuration_states)
                epg_configuration_states = rf_refocusing_pulse.forward(epg_configuration_states)
                epg_configuration_states = gradient_dephasing.forward(epg_configuration_states)
                epg_configuration_states = te_half_relaxation.forward(epg_configuration_states)
                signal[..., i + j * n_refocusing_pulses] = epg_configuration_states[..., 0, 0]

            epg_configuration_states = tr_relaxation.forward(epg_configuration_states)

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
