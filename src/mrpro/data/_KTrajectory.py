"""K-space trajectory classes."""

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

import dataclasses
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch

from mrpro.data._KHeader import KHeader


class KTrajectory(ABC):
    """Base class for k-space trajectories."""

    @abstractmethod
    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the trajectory for the given header.

        The shape should broadcastable to (prod(all_other_dimensions),
        3, k2, k1, k0)
        """
        ...


class KTrajectoryRpe(KTrajectory):
    """Radial phase encoding trajectory.

    Frequency encoding along kx is carried out in a standard Cartesian way. The phase encoding points along ky and kz
    are positioned along radial lines. More details can be found in: https://doi.org/10.1002/mrm.22102  or here
    https://doi.org/10.1118/1.4890095 (open access).

    Parameters
    ----------
    angle
        angle in rad between two radial phase encoding lines
    shift_between_rpe_lines
        shift between radial phase encoding lines along the radial direction.
        See _apply_shifts_between_rpe_lines() for more details
    rad_us_factor
        undersampling factor along each radial phase encoding line
    """

    def __init__(
        self,
        angle: float,
        shift_between_rpe_lines: torch.Tensor = torch.tensor([0, 2, 1, 3]),
        rad_us_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.angle: float = angle
        self.shift_between_rpe_lines: torch.Tensor = shift_between_rpe_lines
        self.rad_us_factor: float = rad_us_factor

    def _apply_shifts_between_rpe_lines(self, krad: torch.Tensor, kang_idx: torch.Tensor):
        """Shift radial phase encoding lines relative to each other.

        E.g. shift_between_rpe_lines = [0, 2, 1, 3] leads to a shift of the 0th line by 0,
        the 1st line by 2 * 1/4 delta_k, the 2nd line by 1 * 1/4 delta_k, the 3rd line by 3 * 1/4 delta_k, the 4th line
        by 0, the 5th line by 2* 1/4 delta_k and so on. The phase encoding points in the k-space center are not shifted.

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
            indices of angles
        """
        delta_shift = 1 / len(self.shift_between_rpe_lines)
        for ind in range(len(self.shift_between_rpe_lines)):
            # curr_angle_idx = np.where(np.mod(kang_idx, len(self.shift_between_rpe_lines)) == ind)[0]
            curr_angle_idx = torch.nonzero(np.mod(kang_idx, len(self.shift_between_rpe_lines)) == ind, as_tuple=True)[0]
            curr_shift = self.shift_between_rpe_lines[ind] * delta_shift
            curr_krad = krad[curr_angle_idx]

            # Do not shift the k-space center
            center_idx = curr_krad == 0
            curr_krad = curr_krad + curr_shift
            curr_krad[center_idx] = 0

            krad[curr_angle_idx] = curr_krad
        return krad

    def _combine_to_3d_traj(self, krad: torch.Tensor, kang: torch.Tensor, kx: torch.Tensor) -> torch.Tensor:
        """Combine k-space points along three directions to 3D tensor.

        Parameters
        ----------
        krad
            k-space points along radial phase encoding lines
        kang
            Angles of phase encoding lines
        kx
            k-space points along readoud

        Returns
        -------
            3D k-space trajectory
        """
        ky = torch.repeat_interleave((krad * torch.cos(kang))[:, None], kx.shape[0], dim=1)
        kz = torch.repeat_interleave((krad * torch.sin(kang))[:, None], kx.shape[0], dim=1)
        kx = torch.repeat_interleave(kx[None, :], ky.shape[0], dim=0)
        return torch.concatenate([kx[:, None, :], ky[:, None, :], kz[:, None, :]], dim=1)

    def calc_traj(self, kheader: KHeader) -> torch.Tensor:
        """Radial phase encoding traejctory.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data

        Returns
        -------
            radial phase encoding trajectory
        """
        # Calculate points along readout
        kx = (kheader.acq_info.number_of_samples - kheader.acq_info.center_sample).to(torch.float32)
        kx *= np.pi / kheader.acq_info.number_of_samples.to(torch.float32)

        # K-space locations along phase encoding lines
        krad = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        krad *= np.pi / kheader.encoding_limits.k1.max

        # Angles of phase encoding lines
        kang_idx = kheader.acq_info.idx.k2 - kheader.encoding_limits.k2.center
        kang = kang_idx * self.angle

        # Shift along radial direction
        krad = self._apply_shifts_between_rpe_lines(krad, kang_idx)

        return self._combine_to_3d_traj(krad, kang, kx)


class KTrajectorySunflowerGoldenRpe(KTrajectoryRpe):
    """Calculate Radial phase encoding trajectory with a sunflower pattern.

    Parameters
    ----------
    KTrajectoryRpe
        Sunflower Golden angle radial phase encoding trajectory
    """

    def __init__(self, rad_us_factor: float) -> None:
        super().__init__(angle=np.pi * 0.618034, rad_us_factor=rad_us_factor)

    def _apply_sunflower_shift_between_rpe_lines(self, krad: torch.Tensor, kang: torch.Tensor, kheader: KHeader):
        """Shift radial phase encoding lines relative to each other.

        The shifts are applied to create a sunflower pattern of k-space points in the ky-kz phase encoding plane.

        Parameters
        ----------
        krad
            k-space positions along each phase encoding line
        kang
            angles of the radial phase encoding lines
        kheader
            MR raw data header (KHeader) containing required meta data
        """
        _, indices = np.unique(kang, return_index=True)
        ang_idx = np.argsort([kang[index] for index in sorted(indices)])
        shift_idx = np.zeros(max(ang_idx) + 1)
        shift_idx[ang_idx] = np.arange(len(ang_idx))

        # Apply sunflower shift
        GoldenCut = 0.5 * (np.sqrt(5) + 1)
        for ind in range(int(len(shift_idx))):
            krad[kheader.acq_info.idx.k2 == ind] = krad[kheader.acq_info.idx.k2 == ind] + 0.5 * (
                2 * ((shift_idx[ind] * GoldenCut) % 1) - 1
            )

        # Set asym k-space point to 0
        krad[kheader.acq_info.idx.k1 == 0] = 0
        return krad

    def calc_traj(self, kheader: KHeader) -> torch.Tensor:
        """Sunflower golden angle radial phase encoding traejctory.

        Parameters
        ----------
        kheader
           MR raw data header (KHeader) containing required meta data.

        Returns
        -------
            sunflower golden angle radial phase encoding trajectory
        """
        # Calculate points along readout
        kx = (kheader.acq_info.number_of_samples - kheader.acq_info.center_sample).to(torch.float32)
        kx *= np.pi / kheader.acq_info.number_of_samples.to(torch.float32)

        # K-space locations along phase encoding lines
        krad = (kheader.acq_info.idx.k1 - kheader.encoding_limits.k1.center).to(torch.float32)
        krad *= np.pi / kheader.encoding_limits.k1.max

        # Angles of phase encoding lines
        kang_idx = kheader.acq_info.idx.k2 - kheader.encoding_limits.k2.center
        kang = (kang_idx * self.angle) % np.pi

        # Shift along radial direction
        krad = self._apply_sunflower_shift_between_rpe_lines(krad, kang, kheader)

        return self._combine_to_3d_traj(krad, kang, kx)


class DummyTrajectory(KTrajectory):
    """Simple Dummy trajectory that returns zeros.

    Shape will not fit to all data. Only used until we implement proper
    trajectories
    """

    @staticmethod
    def _get_shape(header: KHeader) -> tuple[int, ...]:
        """Get the shape of a basic dummy trajectory for the given header.

        Assumes fully sampled data. Do not use outside of testing.
        """
        limits = header.encoding_limits
        other_dim = np.prod(
            [
                getattr(limits, field.name).length
                for field in dataclasses.fields(limits)
                if field.name not in ('k0', 'k1', 'k2', 'segment')
            ]
        )
        shape = (
            other_dim,
            3,
            limits.k2.length,
            limits.k1.length,
            limits.k0.length,
        )
        return shape

    def calc_traj(self, header: KHeader) -> torch.Tensor:
        """Calculate the trajectory for the given header.

        The shape should broadcastable to (prod(all_other_dimensions),
        3, k2, k1, k0)
        """
        return torch.zeros(self._get_shape(header))
