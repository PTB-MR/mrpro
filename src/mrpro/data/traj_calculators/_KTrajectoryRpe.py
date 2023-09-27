"""Radial phase encoding (RPE) trajectory class."""

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

from __future__ import annotations

import torch
from einops import rearrange
from einops import repeat

from mrpro.data._KHeader import KHeader
from mrpro.data.traj_calculators._KTrajectory import KTrajectory


class KTrajectoryRpe(KTrajectory):
    """Radial phase encoding trajectory.

    Frequency encoding along kx is carried out in a standard Cartesian way. The phase encoding points along ky and kz
    are positioned along radial lines. More details can be found in: https://doi.org/10.1002/mrm.22102 and
    https://doi.org/10.1118/1.4890095 (open access).

    Parameters
    ----------
    angle
        angle in rad between two radial phase encoding lines
    shift_between_rpe_lines
        shift between radial phase encoding lines along the radial direction.
        See _apply_shifts_between_rpe_lines() for more details
    """

    def __init__(
        self,
        angle: float,
        shift_between_rpe_lines: torch.Tensor = torch.tensor([0, 0.5, 0.25, 0.75]),
    ) -> None:
        super().__init__()
        self.angle: float = angle
        self.shift_between_rpe_lines: torch.Tensor = shift_between_rpe_lines

    def _apply_shifts_between_rpe_lines(self, krad: torch.Tensor, kang_idx: torch.Tensor):
        """Shift radial phase encoding lines relative to each other.

        Example: shift_between_rpe_lines = [0, 0.5, 0.25, 0.75] leads to a shift of the 0th line by 0,
        the 1st line by 0.5*delta_k, the 2nd line by 0.25*delta_k, the 3rd line by 0.75*delta_k, the 4th line
        by 0, the 5th line by 0.5*delta_k and so on. Phase encoding points in k-space center are not shifted.

        Line #          k-space points before shift             k-space points after shift
        0               +    +    +    +    +    +    +         +    +    +    +    +    +    +
        1               +    +    +    +    +    +    +           +    +    +  +      +    +    +
        2               +    +    +    +    +    +    +          +    +    +   +     +    +    +
        3               +    +    +    +    +    +    +            +    +    + +       +    +    +
        4               +    +    +    +    +    +    +         +    +    +    +    +    +    +
        5               +    +    +    +    +    +    +           +    +    +  +      +    +    +

        More information can be found here: https://doi.org/10.1002/mrm.22446

        Parameters
        ----------
        krad
            k-space positions along each phase encoding line
        kang_idx
            indices of angles to be used for shift calculation
        """
        for ind, shift in enumerate(self.shift_between_rpe_lines):
            curr_angle_idx = torch.nonzero(
                torch.fmod(kang_idx, len(self.shift_between_rpe_lines)) == ind, as_tuple=True
            )
            curr_krad = krad[curr_angle_idx]

            # Do not shift the k-space center
            curr_krad += shift * (curr_krad != 0)

            krad[curr_angle_idx] = curr_krad
        return krad

    def _combine_to_3d_traj(self, krad: torch.Tensor, kang: torch.Tensor, k0: torch.Tensor) -> torch.Tensor:
        """Combine k-space points along three directions to 3D tensor.

        Parameters
        ----------
        krad
            k-space points along radial phase encoding lines
        kang
            Angles of phase encoding lines
        k0
            k-space points along readoud

        Returns
        -------
            3D k-space trajectory
        """
        k1 = repeat(krad * torch.cos(kang), '...->... k0', k0=k0.shape[-1])
        k2 = repeat(krad * torch.sin(kang), '...->... k0', k0=k0.shape[-1])
        k0 = k0.expand(k1.shape)
        return rearrange([k0, k1, k2], 'dir other k2 k1 k0->other dir k2 k1 k0')

    def _k0_traj(self, num_samples: torch.Tensor, center_sample: torch.Tensor):
        """Calculate the trajectory along one readout (k0 dimension).

        Parameters
        ----------
        num_samples
            Number of k-space samples along each readout
        center_sample
            K-space center sample position along readout

        Returns
        -------
            Trajectory along ONE readout

        Raises
        ------
        ValueError
            Number of samples have to be the same for each readout
        ValueError
            Center sample has to be the same for each readout
        """
        # Verify that each readout has the same number of samples and same center sample
        if len(torch.unique(num_samples)) > 1:
            raise ValueError('RPE trajectory can only be calculated if each acquisition has the same number of samples')
        if len(torch.unique(center_sample)) > 1:
            raise ValueError('RPE trajectory can only be calculated if each acquisition has the same center sample')

        # Calculate points along readout
        nk0 = int(num_samples[0, 0, 0])
        k0 = torch.linspace(0, nk0 - 1, nk0, dtype=torch.float32) - center_sample[0, 0, 0]
        k0 *= 2 * torch.pi / nk0
        return k0

    def calc_traj(self, kheader: KHeader) -> torch.Tensor:
        """Calculate radial phase encoding trajectory for given KHeader.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
            radial phase encoding trajectory for given KHeader
        """
        # Calculate points along readout
        k0 = self._k0_traj(kheader.acq_info.number_of_samples, kheader.acq_info.center_sample)

        # Angles of phase encoding lines
        kang = kheader.acq_info.idx.k2 * self.angle

        # K-space locations along phase encoding lines
        krad = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        krad = self._apply_shifts_between_rpe_lines(krad, kheader.acq_info.idx.k2)
        krad *= 2 * torch.pi / kheader.encoding_limits.k1.max

        return self._combine_to_3d_traj(krad, kang, k0)
