"""Returns the trajectory saved in an ISMRMRD raw data file."""

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

import ismrmrd
import torch

from mrpro.data import KTrajectoryRawShape


class KTrajectoryIsmrmrd:
    """Get trajectory in ISMRMRD raw data file.

    The trajectory in the ISMRMRD raw data file is read out. Information
    on the ISMRMRD trajectory can be found here:
    https://ismrmrd.readthedocs.io/en/latest/mrd_raw_data.html#k-space-trajectory

    The value range of the trajectory in the ISMRMRD file is not well defined. Here we simple normalize everything
    based on the highest value and ensure it is within [-pi, pi]. The trajectory is in the shape of the unsorted
    raw data.
    """

    def __init__(self):
        pass

    def __call__(self, acquisitions: list[ismrmrd.Acquisition]) -> KTrajectoryRawShape:
        """Read out the trajectory from the ISMRMRD data file.

        Parameters
        ----------
        acquisitions:
            list of ismrmrd acquisistions to read from. Needs at least one acquisition.

        Returns
        -------
            trajectory in the shape of the original raw data.
        """
        # Read out the trajectory
        ktraj_mrd = torch.stack([torch.as_tensor(acq.traj, dtype=torch.float32) for acq in acquisitions])

        if ktraj_mrd.numel() == 0:
            raise ValueError('No trajectory information available in the acquisitions.')

        if ktraj_mrd.shape[2] == 2:
            ktraj = KTrajectoryRawShape(
                kz=torch.zeros_like(ktraj_mrd[..., 1]),
                ky=ktraj_mrd[..., 1],
                kx=ktraj_mrd[..., 0],
            )
        else:
            ktraj = KTrajectoryRawShape(kz=ktraj_mrd[..., 2], ky=ktraj_mrd[..., 1], kx=ktraj_mrd[..., 0])

        return ktraj
