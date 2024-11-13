"""MR noise measurements class."""

import dataclasses
from collections.abc import Callable
from pathlib import Path

import ismrmrd
import torch
from einops import repeat
from typing_extensions import Self

from mrpro.data.acq_filters import is_noise_acquisition
from mrpro.data.MoveDataMixin import MoveDataMixin


@dataclasses.dataclass(slots=True, frozen=True)
class KNoise(MoveDataMixin):
    """MR raw data / k-space data class for noise measurements."""

    data: torch.Tensor
    """K-space data of noise measurements. Shape (...other coils k2 k1 k0)"""

    @classmethod
    def from_file(
        cls, filename: str | Path, dataset_idx: int = -1, acquisition_filter_criterion: Callable = is_noise_acquisition
    ) -> Self:
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
        noise_data = repeat(noise_data, '... coils k0->... coils k2 k1 k0', k1=1, k2=1)

        return cls(noise_data)

    def __repr__(self):
        """Representation method for KNoise class."""
        try:
            device = str(self.device)
        except RuntimeError:
            device = 'mixed'
        name = type(self).__name__
        return f'{name} with shape: {list(self.data.shape)!s} and dtype {self.data.dtype}\nDevice: {device}.'
