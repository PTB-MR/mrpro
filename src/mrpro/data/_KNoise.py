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
from pathlib import Path
from typing import Self

import ismrmrd
import torch
from einops import rearrange

from mrpro.data.enums import AcqFlags


@dataclasses.dataclass(slots=True, frozen=True)
class KNoise:
    """MR raw data / k-space data class for noise measurements.

    Attributes
    ----------
    data
        k-space data of noise measurements as complex tensor
    """

    data: torch.Tensor

    @classmethod
    def from_file(cls, filename: str | Path, dataset_idx: int = -1) -> KNoise:
        """Load noise measurements from ISMRMRD file.

        Parameters
        ----------
        filename
            Path to the ISMRMRD file
        dataset_idx
            Index of the dataset to load (converter creates dataset, dataset_1, ...), default is -1 (last)
        """
        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            ds = file[list(file.keys())[dataset_idx]]
            acquisitions = ds.acquisitions[:]

        # Read out noise measurements
        acquisitions = list(filter(lambda acq: (AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value & acq.flags), acquisitions))
        if len(acquisitions) == 0:
            raise ValueError(f'No noise measurements found in {filename}')
        noise_data = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        # Reshape to standard dimensions
        noise_data = rearrange(noise_data, 'other coils (k2 k1 k0)->other coils k2 k1 k0', k1=1, k2=1)

        return cls(noise_data)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: None | torch.dtype = None,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self:
        """Perform dtype and/or device conversion of data.

        Parameters
        ----------
        device
            The destination device. Defaults to the current device.
        dtype
            Dtype of the k-space data, can only be torch.complex64 or torch.complex128.
            The dtype of the trajectory (torch.float32 or torch.float64) is then inferred from this.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        copy
            If True a new Tensor is created even when the Tensor already matches the desired conversion.
        memory_format
            The desired memory format of returned Tensor.
        """
        return type(self)(
            data=self.data.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            )
        )

    def cuda(
        self,
        device: torch.device | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> Self:
        """Create copy of object with trajectory and data in CUDA memory.

        Parameters
        ----------
        device
            The destination GPU device. Defaults to the current CUDA device.
        non_blocking
            If True and the source is in pinned memory, the copy will be asynchronous with respect to the host.
            Otherwise, the argument has no effect.
        memory_format
            The desired memory format of returned Tensor.
        """
        return type(self)(
            data=self.data.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format),  # type: ignore [call-arg]
        )

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> Self:
        """Create copy of object in CPU memory.

        Parameters
        ----------
        memory_format
            The desired memory format of returned Tensor.
        """
        return type(self)(
            data=self.data.cpu(memory_format=memory_format),  # type: ignore [call-arg]
        )
