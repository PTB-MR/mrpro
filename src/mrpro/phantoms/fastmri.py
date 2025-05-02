"""FastMRI dataset."""

from pathlib import Path

import h5py
import ismrmrd.xsd
import numpy as np
import torch
from einops import rearrange

import mrpro
from mrpro.data.KData import KData
from mrpro.utils.typing import FileOrPath


class FastMRIDataset(torch.utils.data.Dataset):
    """FastMRI T(Training) Dataset.

    This dataset returns KData objects for single slices of the FastMRI brain or knee dataset.
    The data has to be downloaded beforehand. See https://fastmri.med.nyu.edu/ for more information.
    """

    def __init__(self, data_path: FileOrPath):
        """Initialize the dataset.

        Parameters
        ----------
        data_path : PathLike
            Path to the data directory.
        """
        self.file_list = list(Path(data_path).rglob('*.h5'))
        slices = []
        self.shapes = []
        for fn in self.file_list:
            with h5py.File(fn, 'r') as file:
                acquisition: str = file.attrs['acquisition']
                if acquisition.startswith('AX'):  # brain
                    slices.append(file['kspace'].shape[1])
                elif acquisition.endswith('FBK'):  # knee
                    slices.append(file['kspace'].shape[0])
                else:
                    raise ValueError(f'Unknown acquisition: {acquisition}')
                self.shapes.append(file['kspace'].shape)
        self._accum_slices = torch.tensor(slices).cumsum(dim=0)

    def __len__(self) -> int:
        """Get length (number of slices) of the dataset."""
        return int(self._accum_slices[-1])

    def __getitem__(self, idx: int) -> KData:
        """Get a single slice."""
        if not -len(self) <= idx < len(self):
            raise IndexError(f'Index {idx} is out of bounds for the dataset of size {len(self)}')
        if idx < 0:
            idx += len(self)
        file_idx = torch.searchsorted(self._accum_slices, idx + 1)
        slice_idx = idx - self._accum_slices[file_idx]
        with h5py.File(self.file_list[file_idx], 'r') as file:
            acquisition: str = file.attrs['acquisition']
            if acquisition.startswith('AX'):  # brain
                data = torch.as_tensor(np.array(file['kspace'][:, slice_idx]))
            elif acquisition.endswith('FBK'):  # knee
                data = torch.as_tensor(np.array(file['kspace'][slice_idx]))
            else:
                raise ValueError(f'Unknown acquisition: {acquisition}')
            data = rearrange(data, 'coil k0 k1 ->1 coil 1 k1 k0')
            n_k1, n_k0 = data.shape[-2:]
            info = mrpro.data.AcqInfo()
            info.idx.k1 = torch.arange(n_k1)[None, None, None, :, None]
            header = mrpro.data.KHeader.from_ismrmrd(
                ismrmrd.xsd.CreateFromDocument(file['ismrmrd_header'][()].decode('utf-8')), info
            )
            traj = mrpro.data.traj_calculators.KTrajectoryCartesian()(
                n_k0=n_k0,
                k0_center=n_k0 // 2,
                k1_idx=info.idx.k1,
                k1_center=n_k1 // 2,
                k2_idx=torch.tensor(0),
                k2_center=0,
            )
            kdata = mrpro.data.KData(
                data=data,
                header=header,
                traj=traj,
            )
            return kdata
