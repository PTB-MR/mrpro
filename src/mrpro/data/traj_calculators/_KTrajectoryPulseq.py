"""K-space trajectory from pulseq seq file."""

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

import numpy as np
import pypulseq
import torch

from mrpro.data.traj_calculators import KTrajectoryCalculator

# TODO: This is work in progress
# copied over from https://github.com/Stef-Martin/MRERecon/blob/ReadData/ReadData.py


class KTrajectoryPulseq(KTrajectoryCalculator):
    """Trajectory from .seq file.

    Parameters
    ----------
    path
        path to .seq file
    """

    def __init__(
        self,
        path: str,  # TODO: should accept a pathlib.Path as well
    ) -> None:
        super().__init__()
        self.path: str = path
        # TODO: Read in pulseq file here to throw error if file is corrupt or does not exist

    # TODO: should be __call__(self, header:Kheader)->KTrajectory
    def calc_traj(self) -> torch.Tensor:
        """Get trajectory for given KHeader.

        Parameters
        ----------
        path:

        Returns
        -------
            phase encoding trajectory for given path
        """
        seq = pypulseq.Sequence()
        seq.read(file_path=self.path)
        k_traj_adc, *_ = seq.calculate_kspacePP()

        kx = k_traj_adc[0]
        ky = k_traj_adc[1]
        kz = k_traj_adc[2]

        # ToDo: get this out of the sequence file
        N_shots = 1
        N_slices = 1
        #####
        num_spirals = N_shots
        num_slices = N_slices
        num_k_per_spiral = int(kx.shape[0] / num_spirals / num_slices)

        # TODO: just use kz,ky,kx to create KTrajectory
        k0 = num_k_per_spiral
        k1 = num_spirals
        k2 = num_slices

        tensor_shape = (3, k2, k1, k0)
        tensor = np.empty(tensor_shape)

        # Create an index grid for broadcasting
        i, j, k = np.indices((num_spirals, num_k_per_spiral, num_slices))

        # Assign values to tensor using broadcasting
        tensor[0, k, i, j] = kx[j]
        tensor[1, k, i, j] = ky[j]
        tensor[2, k, i, j] = kz[j]

        ktraj = torch.from_numpy(tensor)
        ktraj = ktraj.unsqueeze(dim=0)
        # TODO: return KTrajectory
        return ktraj
