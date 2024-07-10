"""MR noise measurements class."""

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

import dataclasses
from collections.abc import Callable
from pathlib import Path

import ismrmrd
import torch
from einops import rearrange

from mrpro.data.acq_filters import is_noise_acquisition
from mrpro.data.MoveDataMixin import MoveDataMixin


@dataclasses.dataclass(slots=True, frozen=True)
class KNoise(MoveDataMixin):
    """MR raw data / k-space data class for noise measurements.

    Attributes
    ----------
    data
        k-space data of noise measurements as complex tensor
    """

    data: torch.Tensor

    @classmethod
    def from_file(
        cls, filename: str | Path, dataset_idx: int = -1, acquisition_filter_criterion: Callable = is_noise_acquisition
    ) -> KNoise:
        """Load noise measurements from ISMRMRD file.

        Parameters
        ----------
        filename
            Path to the ISMRMRD file
        dataset_idx
            Index of the dataset to load (converter creates dataset, dataset_1, ...)
        acquisition_filter_criterion
            function which returns True if an acquisition should be included in KNoise
        """
        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            ds = file[list(file.keys())[dataset_idx]]
            acquisitions = ds.acquisitions[:]

        # Read out noise measurements
        acquisitions = [acq for acq in acquisitions if acquisition_filter_criterion(acq)]
        if len(acquisitions) == 0:
            raise ValueError(f'No noise measurements found in {filename}')
        noise_data = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        # Reshape to standard dimensions
        noise_data = rearrange(noise_data, 'other coils (k2 k1 k0)->other coils k2 k1 k0', k1=1, k2=1)

        return cls(noise_data)
     
    def __repr__(self):
        return f"KNoise with shape: {str(list(self.data.shape))}."