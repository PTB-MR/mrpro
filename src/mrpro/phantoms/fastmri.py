"""FastMRI dataset."""

from collections.abc import Callable
from os import PathLike
from pathlib import Path

import h5py
import ismrmrd.xsd
import numpy as np
import torch
from einops import rearrange

import mrpro
from mrpro.algorithms.csm import walsh
from mrpro.data.KData import KData
from mrpro.utils.interpolate import apply_lowres
from mrpro.utils.pad_or_crop import pad_or_crop


class FastMRIKDataDataset(torch.utils.data.Dataset):
    """FastMRI KData Dataset.

    This dataset returns KData objects for single slices of the FastMRI brain or knee dataset.
    The data has to be downloaded beforehand. See https://fastmri.med.nyu.edu/ for more information.
    """

    def __init__(self, data_path: PathLike | str):
        """Initialize the dataset.

        Parameters
        ----------
        data_path : PathLike
            Path to the data directory.
        """
        self._filenames = list(Path(data_path).rglob('*.h5'))
        slices = []
        for fn in self._filenames:
            with h5py.File(fn, 'r') as file:
                acquisition: str = file.attrs['acquisition']
                if acquisition.startswith('AX'):  # brain
                    slices.append(file['kspace'].shape[1])
                elif acquisition.endswith('FBK'):  # knee
                    slices.append(file['kspace'].shape[0])
                else:
                    raise ValueError(f'Unknown acquisition: {acquisition}')
        self._accum_slices = torch.tensor(slices).cumsum(dim=0)

    def __len__(self) -> int:
        """Get length (number of slices) of the dataset."""
        if len(self._accum_slices) == 0:
            return 0
        return int(self._accum_slices[-1])

    def __getitem__(self, idx: int) -> KData:
        """Get a single slice."""
        if not -len(self) <= idx < len(self):
            raise IndexError(f'Index {idx} is out of bounds for the dataset of size {len(self)}')
        if idx < 0:
            idx += len(self)
        file_idx = torch.searchsorted(self._accum_slices, idx + 1)
        slice_idx = idx - self._accum_slices[file_idx]
        with h5py.File(self._filenames[file_idx], 'r') as file:
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


class FastMRIImageDataset(torch.utils.data.Dataset):
    """FastMRI Image Dataset.

    This dataset returns image tensors for single slices of the FastMRI brain or knee dataset.
    It filteres and resamples the files such that the returned images have a consistent shape
    of 320x320 before augmentations.

    The returned images are complex valued and will have shape ``(1, 1, 1, 320, 320)`` if coil combined or
    ``(1, n_coils, 1, 320, 320)`` otherwise, with `n_coils` 15 for knee data and 16 for brain data.

    The data has to be downloaded beforehand. See https://fastmri.med.nyu.edu/ for more information.
    """

    def __init__(
        self,
        data_path: PathLike | str,
        coil_combine: bool = False,
        augment: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    ):
        """Initialize the dataset.

        Parameters
        ----------
        data_path : PathLike
            Path to the data directory.
        coil_combine : bool
            Whether to perform coil combination sensitivity maps obtained using the Walsh method.
            Note that this is **not** comonly used as the target for FastMRI challenges. Instead,
            as target the RSS combination of the coil images is used.
        augment
            Augmentation function. Will be called with the image and the index of the slices.
            If `coil_combine` is `True`, the function will be called with the complex valued coil combined image
            with shape (1, 320, 320)
            otherwise with the complex valued coil images
            with shape (n_coils, 320, 320).
            `None` means no augmentation.
        """
        slices = []
        self._filenames = []
        for fn in Path(data_path).rglob('*.h5'):
            with h5py.File(fn, 'r') as file:
                acquisition: str = file.attrs['acquisition']
                shape = file['kspace'].shape
                header = mrpro.data.KHeader.from_ismrmrd(
                    ismrmrd.xsd.CreateFromDocument(file['ismrmrd_header'][()].decode('utf-8')), mrpro.data.AcqInfo()
                )
                if acquisition.startswith('AX'):  # brain
                    n_coils, n_slices = shape[:2]
                    if (
                        (n_coils != 16 and not coil_combine)
                        or round(header.recon_fov.y, 2) != 0.22
                        or header.recon_matrix.x not in [320, 384]
                    ):
                        continue
                elif acquisition.endswith('FBK'):
                    n_slices, n_coils = shape[:2]  # different order for knee
                    if (
                        (n_coils != 15 and not coil_combine)
                        or header.recon_matrix.x != 320
                        or round(header.recon_fov.y, 2) != 0.14
                    ):
                        continue
                else:
                    raise ValueError(f'Unknown acquisition: {acquisition}')
                slices.append(n_slices)
                self._filenames.append(fn)
        self._accum_slices = torch.tensor(slices).cumsum(dim=0)
        self._coil_combine = coil_combine
        self.augment = augment

    def __len__(self) -> int:
        """Get length (number of slices) of the dataset."""
        if len(self._accum_slices) == 0:
            return 0
        return int(self._accum_slices[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single slice."""
        if not -len(self) <= idx < len(self):
            raise IndexError(f'Index {idx} is out of bounds for the dataset of size {len(self)}')
        if idx < 0:
            idx += len(self)
        file_idx = torch.searchsorted(self._accum_slices, idx + 1)
        slice_idx = idx - self._accum_slices[file_idx]
        with h5py.File(self._filenames[file_idx], 'r') as file:
            acquisition: str = file.attrs['acquisition']
            if acquisition.startswith('AX'):  # brain
                data = torch.as_tensor(np.array(file['kspace'][:, slice_idx]))
            elif acquisition.endswith('FBK'):  # knee
                data = torch.as_tensor(np.array(file['kspace'][slice_idx]))
            else:
                raise ValueError(f'Unknown acquisition: {acquisition}')
            header = mrpro.data.KHeader.from_ismrmrd(
                ismrmrd.xsd.CreateFromDocument(file['ismrmrd_header'][()].decode('utf-8')), mrpro.data.AcqInfo()
            )
            # recon matrix can be larger than 320, but FOV is always constant. So we crop in k-space.
            data = pad_or_crop(
                data,
                (int(data.shape[-2] * 320 / header.recon_matrix.y), int(data.shape[-1] * 320 / header.recon_matrix.y)),
                dim=(-2, -1),
            )
            # we use an fft instead of ifft to already flip the image
            img = torch.fft.fftshift(torch.fft.fft2(torch.fft.fftshift(data, dim=(-2, -1))), dim=(-2, -1))
            # finally, we crop to the new recon_fov of 320x320 to remove oversampling
            img = pad_or_crop(img, (320, 320), dim=(-2, -1))
            if self._coil_combine:
                csm = apply_lowres(
                    lambda x: walsh(x.unsqueeze(1), smoothing_width=3).squeeze(1), (32, 32), dim=(-2, -1)
                )(img)
                img = (img * csm.conj()).sum(dim=0, keepdim=True)
            if self.augment is not None:
                img = self.augment(img, idx)
            return rearrange(img, 'coils y x -> 1 coils 1 y x')
