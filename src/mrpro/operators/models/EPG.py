"""Extended phase graph (EPG) signal models."""

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
from collections.abc import Sequence

import numpy as np
import torch

from mrpro.operators.SignalModel import SignalModel


class EpgRfPulse:
    """Mixing of EPG configuration states due to RF pulse."""

    def __init__(
        self,
        flip_angle: float | torch.Tensor,
        phase: float | torch.Tensor,
        b1_scaling_factor: torch.Tensor | None = None,
    ):
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
        flip_angle = torch.as_tensor(flip_angle)
        phase = torch.as_tensor(phase)
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
    """Relaxation (i.e. reduction of population levels) of EPG states."""

    def __init__(
        self, relaxation_time: float | torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, t1_recovery: bool = True
    ):
        """Relaxation of EPG states.

        Parameters
        ----------
        relaxation_time
            relaxation time
        t1
            longitudinal relaxation time
        t2
            transversal relaxation time
        t1_recovery
            recovery of longitudinal EPG states
        """
        relaxation_time = torch.as_tensor(relaxation_time)
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


class EpgMrfFispWithPreparation(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for MRF with preparation pulses (e.g. T2-prep) using FISP sequence.

    The data acquisition is described as multiple blocks. At the beginning of each block is a preparation pulse,
    followed by a certain number of RF pulses for data acquisition. After each block is a delay without data
    acquisition. For data acquisition a fast imaging with steady-state free precession (FISP) sequence is used.

    Classic MRF with a single inversion-pulse at the beginning followed by a train of RF pulses with different flip
    angles as described in Ma, D. et al. Magnetic resonance fingerprinting. Nature 495, 187-192 (2013)
    [http://dx.doi.org/10.1038/nature11971] can be simulated as a single block with e.g.:

    inv_prep_ti = 20 # 20 ms delay after inversion pulse before data acquisition starts
    te_prep_te = None # No T2-preparation pulse
    n_rf_pulses_per_block = None # A single block so all RF pulses defined by flip_angles/rf_phases will be carried out
    # delay_after_block can be left as default because it is note used for a single block

    Cardiac MRF where a different preparation is done in each cardiac cycle followed by fixed number of RF pulses is
    described in Hamilton, J. I. et al. MR fingerprinting for rapid quantification of myocardial T1 , T2 , and proton
    spin density. Magn. Reson. Med. 77, 1446-1458 (2017) [http://doi.wiley.com/10.1002/mrm.26668]. It is a four-fold
    repetition of

                Block 0                   Block 1                   Block 2                     Block 3
       R-peak                   R-peak                    R-peak                    R-peak                    R-peak
    ---|-------------------------|-------------------------|-------------------------|-------------------------|-----

            [INV TI=20ms][ACQ]                     [ACQ]     [T2-prep TE=40ms][ACQ]    [T2-prep TE=80ms][ACQ]

    can be simulated as:

    inv_prep_ti = [20,None,None,None]*4 # 20 ms delay after inversion pulse in block 0
    te_prep_te = [None,None,40,80]*4 # T2-preparation pulse with TE = 40 and 80 in block 2 and 3, respectively
    n_rf_pulses_per_block = 48 # 48 RF pulses in each block
    # Cardiac trigger delay is 700 ms. The delay between blocks is smaller because the duration of the preparation
    # pulses needs to be taken into consideration
    delay_after_block = [700, 660, 620, 680]*4

    """

    def __init__(
        self,
        flip_angles: float | torch.Tensor,
        rf_phases: float | torch.Tensor,
        te: float | torch.Tensor,
        tr: float | torch.Tensor,
        inv_prep_ti: None | float | Sequence[float | None] = None,
        t2_prep_te: None | float | Sequence[float | None] = None,
        n_rf_pulses_per_block: None | int | Sequence[int] = None,
        delay_after_block: float | Sequence[float] = 0.0,
        max_n_configuration_states: int | float = torch.inf,
    ):
        """FISP MRF with multiple preparation pulses.

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
        inv_prep_ti
            TI after inversion pulse, use None for no inversion pulse
        t2_prep_te
            TE of T2-preparation pulse, use None for no T2-preparation pulse
        n_rf_pulses_per_block
            number of RF acquisition pulses in each block
        delay_after_block
            delay after each block
        max_n_configuration_states
            maximum number of configuration states to be considered. Default means all configuration states are
            considered.
        """
        super().__init__()

        # convert all parameters to tensors
        flip_angles = torch.as_tensor(flip_angles)
        rf_phases = torch.as_tensor(rf_phases)
        te = torch.as_tensor(te)
        tr = torch.as_tensor(tr)

        # need to broadcast here because we are looping over these parameters in forward()
        try:
            (flip_angles, rf_phases, te, tr) = torch.broadcast_tensors(flip_angles, rf_phases, te, tr)
        except RuntimeError:
            # Not broadcastable
            raise ValueError(
                f'Shapes of flip_angles ({flip_angles.shape}), rf_phases ({rf_phases.shape}), te ({te.shape}) and '  # type: ignore [union-attr]
                f'tr ({tr.shape}) do not match and also cannot be broadcasted.',
            ) from None

        # nn.Parameters allow for grad calculation
        self.flip_angles = torch.nn.Parameter(flip_angles, requires_grad=flip_angles.requires_grad)  # type: ignore [arg-type, union-attr]
        self.rf_phases = torch.nn.Parameter(rf_phases, requires_grad=rf_phases.requires_grad)  # type: ignore [arg-type, union-attr]
        self.te = torch.nn.Parameter(te, requires_grad=te.requires_grad)  # type: ignore [arg-type, union-attr]
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)  # type: ignore [arg-type, union-attr]

        self.max_n_configuration_states = max_n_configuration_states

        # convert all block parameters to Sequence[float | int] of same length
        self.inv_prep_ti = [inv_prep_ti] if inv_prep_ti is None or isinstance(inv_prep_ti, int | float) else inv_prep_ti
        self.t2_prep_te = [t2_prep_te] if t2_prep_te is None or isinstance(t2_prep_te, int | float) else t2_prep_te
        if n_rf_pulses_per_block is None:
            n_rf_pulses_per_block = int(self.flip_angles.shape[0])
        self.n_rf_pulses_per_block = (
            [n_rf_pulses_per_block] if isinstance(n_rf_pulses_per_block, int) else n_rf_pulses_per_block
        )
        self.delay_after_block = (
            (delay_after_block,) if isinstance(delay_after_block, int | float) else delay_after_block
        )

        n_of_blocks = max(
            len(self.inv_prep_ti), len(self.t2_prep_te), len(self.n_rf_pulses_per_block), len(self.delay_after_block)
        )

        for parameter in [self.inv_prep_ti, self.t2_prep_te, self.n_rf_pulses_per_block, self.delay_after_block]:
            if len(parameter) > 1 and len(parameter) != n_of_blocks:
                raise ValueError(
                    f'All parameters need to be of same length: inv_prep_ti: {len(self.inv_prep_ti)}, '
                    f't2_prep_te: {len(self.t2_prep_te)}, '
                    f'n_rf_pulses_per_block: {len(self.n_rf_pulses_per_block)}, '
                    f'delay_after_block: {len(self.delay_after_block)}'
                )

        if len(self.inv_prep_ti) == 1:
            self.inv_prep_ti = [self.inv_prep_ti[0] for _ in range(n_of_blocks)]
        if len(self.t2_prep_te) == 1:
            self.t2_prep_te = [self.t2_prep_te[0] for _ in range(n_of_blocks)]
        if len(self.n_rf_pulses_per_block) == 1:
            self.n_rf_pulses_per_block = [self.n_rf_pulses_per_block[0] for _ in range(n_of_blocks)]
        if len(self.delay_after_block) == 1:
            self.delay_after_block = [self.delay_after_block[0] for _ in range(n_of_blocks)]

        # only one preparation pulse is possible per block
        for inv_prep_ti, t2_prep_te in zip(self.inv_prep_ti, self.t2_prep_te, strict=False):
            if inv_prep_ti is not None and t2_prep_te is not None:
                raise ValueError('Only one preparation pulse allowed per block.')

        # sum of the Rf pulses in each block has to match the total number of rf pulses
        if np.sum(self.n_rf_pulses_per_block) != self.flip_angles.shape[0]:
            raise ValueError(
                f'Total number of RF pulses ({np.sum(self.n_rf_pulses_per_block)}) need to be the same as the number '
                f'of defined flip_angles ({self.flip_angles.shape[0]}).'
            )

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

        # Start in equilibrium state
        epg_configuration_states[..., :, 0] = torch.tensor((0.0, 0, 1.0))

        for block_idx, inv_prep_ti, t2_prep_te, n_rf_pulses_per_block, delay_after_block in zip(
            range(len(self.inv_prep_ti)),
            self.inv_prep_ti,
            self.t2_prep_te,
            self.n_rf_pulses_per_block,
            self.delay_after_block,
            strict=False,
        ):
            # Preparation pulse
            if inv_prep_ti is not None:
                # 180° inversion pulse -> relaxation
                epg_configuration_states = EpgRfPulse(torch.pi, 0).apply(epg_configuration_states)
                epg_configuration_states = EpgRelaxation(inv_prep_ti, t1, t2).apply(epg_configuration_states)
            elif t2_prep_te is not None:
                # 90° pulse -> relaxation during TE/2 -> 180° pulse -> relaxation during TE/2 -> -90° pulse
                epg_configuration_states = EpgRfPulse(torch.pi / 2, 0).apply(epg_configuration_states)
                epg_configuration_states = EpgRelaxation(t2_prep_te / 2, t1, t2).apply(epg_configuration_states)
                epg_configuration_states = EpgRfPulse(torch.pi, torch.pi / 2).apply(epg_configuration_states)
                epg_configuration_states = EpgRelaxation(t2_prep_te / 2, t1, t2).apply(epg_configuration_states)
                epg_configuration_states = EpgRfPulse(torch.pi / 2, -torch.pi).apply(epg_configuration_states)
                # Spoiler
                gradient_dephasing = EpgGradient(epg_configuration_states.shape[-1] >= self.max_n_configuration_states)
                epg_configuration_states = gradient_dephasing.apply(epg_configuration_states)

            # RF-pulse -> relaxation during TE -> get signal -> gradient -> relaxation during TR-TE
            last_idx_of_previous_block = np.int64(np.sum(self.n_rf_pulses_per_block[:block_idx]))
            for idx_in_block in range(n_rf_pulses_per_block):
                idx_in_total_acq = last_idx_of_previous_block + idx_in_block
                rf_pulse = EpgRfPulse(flip_angles[idx_in_total_acq, ...], rf_phases[idx_in_total_acq, ...])
                te_relaxation = EpgRelaxation(te[idx_in_total_acq, ...], t1, t2)
                tr_relaxation = EpgRelaxation((tr - te)[idx_in_total_acq, ...], t1, t2)
                if idx_in_total_acq == 0 or epg_configuration_states.shape[0] >= self.max_n_configuration_states:
                    gradient_dephasing = EpgGradient(
                        epg_configuration_states.shape[-1] >= self.max_n_configuration_states
                    )

                epg_configuration_states = rf_pulse.apply(epg_configuration_states)
                epg_configuration_states = te_relaxation.apply(epg_configuration_states)
                signal[idx_in_total_acq, ...] = m0 * epg_configuration_states[..., 0, 0]
                epg_configuration_states = gradient_dephasing.apply(epg_configuration_states)
                epg_configuration_states = tr_relaxation.apply(epg_configuration_states)

            # Time of no acquisition between blocks
            if delay_after_block > 0:
                epg_configuration_states = EpgRelaxation(delay_after_block, t1, t2).apply(epg_configuration_states)

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
            repetition time between echo trains. Default corresponds to single echo train.
            with shape (number of echo trains,)
        max_n_configuration_states
            maximum number of configuration states to be considered. Default means all configuration states are
            considered.
        """
        super().__init__()

        # convert all parameters to tensors
        flip_angles = torch.as_tensor(flip_angles)
        rf_phases = torch.as_tensor(rf_phases)
        te = torch.as_tensor(te)
        tr = torch.atleast_1d(torch.as_tensor(tr))

        # need to broadcast here because we are looping over these parameters in forward()
        try:
            (flip_angles, rf_phases, te) = torch.broadcast_tensors(flip_angles, rf_phases, te)
        except RuntimeError:
            # Not broadcastable
            raise ValueError(
                f'Shapes of flip_angles ({flip_angles.shape}), rf_phases ({rf_phases.shape}) and te ({te.shape}) do '  # type: ignore [union-attr]
                'not match and also cannot be broadcasted.',
            ) from None

        # nn.Parameters allow for grad calculation
        self.flip_angles = torch.nn.Parameter(flip_angles, requires_grad=flip_angles.requires_grad)  # type: ignore [arg-type, union-attr]
        self.rf_phases = torch.nn.Parameter(rf_phases, requires_grad=rf_phases.requires_grad)  # type: ignore [arg-type, union-attr]
        self.te = torch.nn.Parameter(te, requires_grad=te.requires_grad)  # type: ignore [arg-type, union-attr]
        self.tr = torch.nn.Parameter(tr, requires_grad=tr.requires_grad)  # type: ignore [arg-type, union-attr]

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

            if j < len(self.tr) - 1:
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
