"""K-space trajectory base class."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
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

from __future__ import annotations

from abc import ABC
from abc import abstractmethod

import torch

from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.data import KTrajectoryRawShape
from mrpro.data.enums import AcqFlags


class KTrajectoryCalculator(ABC):
    """Base class for k-space trajectories."""

    @abstractmethod
    def __call__(self, header: KHeader) -> KTrajectory | KTrajectoryRawShape:
        """Calculate the trajectory for given KHeader.

        The shapes of kz, ky and kx of the calculated trajectory must be
        broadcastable to (prod(all_other_dimensions), k2, k1, k0).
        """
        ...

    def _kfreq(self, kheader: KHeader) -> torch.Tensor:
        """Calculate the trajectory along one readout (k0 dimension).

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data

        Returns
        -------
            trajectory along ONE readout

        Raises
        ------
        ValueError
            Number of samples have to be the same for each readout
        """
        n_samples = torch.unique(kheader.acq_info.number_of_samples)
        center_sample = kheader.acq_info.center_sample
        if len(n_samples) > 1:
            raise ValueError('Trajectory can only be calculated if each acquisition has the same number of samples')
        n_k0 = int(n_samples.item())

        # Data can be obtained with standard or reversed readout (e.g. bipolar readout).
        k0 = torch.linspace(0, n_k0 - 1, n_k0, dtype=torch.float32) - center_sample
        # Data can be obtained with standard or reversed readout (e.g. bipolar readout).
        reversed_readout_mask = (kheader.acq_info.flags[..., 0] & AcqFlags.ACQ_IS_REVERSE.value).bool()
        k0[reversed_readout_mask, :] = torch.flip(k0[reversed_readout_mask, :], (-1,))
        return k0


class DummyTrajectory(KTrajectoryCalculator):
    """Simple Dummy trajectory that returns zeros.

    Shape will fit to all data. Only used as dummy for testing.
    """

    def __call__(self, header: KHeader) -> KTrajectory:  # noqa: ARG002
        """Calculate dummy trajectory."""
        kx = torch.zeros(1, 1, 1, 1)
        ky = kz = torch.zeros(1, 1, 1, 1)
        return KTrajectory(kz, ky, kx)
