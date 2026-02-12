"""FastMRI dataset."""

from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from warnings import warn

import h5py
import ismrmrd.xsd
import numpy as np
import torch
from einops import rearrange

from mr2.algorithms.csm.inati import inati
from mr2.data.AcqInfo import AcqInfo
from mr2.data.KData import KData
from mr2.data.KHeader import KHeader
from mr2.data.traj_calculators.KTrajectoryCartesian import KTrajectoryCartesian
from mr2.utils.interpolate import apply_lowres
from mr2.utils.pad_or_crop import pad_or_crop
from mr2.utils.reshape import unsqueeze_left


class FastMRIKDataDataset(torch.utils.data.Dataset):
    """FastMRI KData Dataset.

    This dataset returns KData objects for single slices or stacks of slices of the FastMRI brain or knee dataset.
    The data has to be downloaded beforehand. See https://fastmri.med.nyu.edu/ for more information.
    """

    def __init__(self, path: PathLike | str | Sequence[PathLike | str], single_slice: bool = True):
        """Initialize the dataset.

        Parameters
        ----------
        path : PathLike
            Either a path to a directory containing the FastMRI data as .h5 files or a sequence of paths of
            individual files.
        single_slice : bool
            Whether to return single slices or stacks of slices.
        """
        self._filenames = []
        slices = []
        for fn in Path(path).rglob('*.h5') if isinstance(path, str | Path | PathLike) else path:
            try:
                with h5py.File(fn, 'r') as file:
                    n_slices = file['kspace'].shape[0]
                    slices.append(n_slices)
                    self._filenames.append(fn)
            except (KeyError, FileNotFoundError, OSError):
                warn(f'Invalid file: {fn}. Skipping.', stacklevel=2)
        self._accum_slices = torch.tensor(slices).cumsum(dim=0) if single_slice else None

    def __len__(self) -> int:
        """Get length (number of slices or stacks of slices) of the dataset."""
        if self._accum_slices is None:
            return len(self._filenames)
        if len(self._accum_slices) == 0:
            return 0
        return int(self._accum_slices[-1])

    def __getitem__(self, idx: int) -> KData:
        """Get a single slice or stack of slices."""
        if not -len(self) <= idx < len(self):
            raise IndexError(f'Index {idx} is out of bounds for the dataset of size {len(self)}')
        if idx < 0:
            idx += len(self)
        if self._accum_slices is None:  # return stack
            file_idx = idx
            slice_idx: int | slice = slice(None)
        else:
            file_idx = int(torch.searchsorted(self._accum_slices, idx + 1))
            slice_idx = int(idx - self._accum_slices[file_idx])
        with h5py.File(self._filenames[file_idx], 'r') as file:
            # data is sometimes zero-padded, we remove the padding
            data = torch.as_tensor(np.array(file['kspace'][slice_idx]))
            data = unsqueeze_left(data, 4 - data.ndim)
            nonzero = data[0, 0, 0, :].abs() > 1e-12
            first, last = nonzero.nonzero().flatten()[[0, -1]]
            data = data[..., first : last + 1]
            data = rearrange(data, 'slices coils k0 k1 -> slices coils 1 k1 k0')
            n_k1, n_k0 = data.shape[-2:]
            info = AcqInfo()
            info.idx.k1 = torch.arange(first, last + 1)[None, None, None, :, None]
            header = KHeader.from_ismrmrd(
                ismrmrd.xsd.CreateFromDocument(file['ismrmrd_header'][()].decode('utf-8')), info
            )
            traj = KTrajectoryCartesian()(
                n_k0=n_k0,
                k0_center=n_k0 // 2,
                k1_idx=info.idx.k1,
                k1_center=first + n_k1 // 2,
                k2_idx=torch.tensor(0),
                k2_center=0,
            )
            kdata = KData(
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
    ``(1, n_coils, 1, 320, 320)`` otherwise.

    The data has to be downloaded beforehand. See https://fastmri.med.nyu.edu/ for more information.
    """

    def __init__(
        self,
        path: str | PathLike | Sequence[str | PathLike],
        coil_combine: bool = False,
        augment: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        allowed_n_coils: Sequence[int] | None = (16, 15),
    ):
        """Initialize the dataset.

        Parameters
        ----------
        path : PathLike
            Either a path to a directory containing the FastMRI data as .h5 files or a sequence of paths of
            individual files.
        coil_combine : bool
            Whether to perform coil combination sensitivity maps obtained using the Inati method.
            Note that this is **not** commonly used as the target for FastMRI challenges. Instead,
            as target the RSS combination of the coil images is used.
        augment
            Augmentation function. Will be called with the image and the index of the slices.
            If `coil_combine` is `True`, the function will be called with the complex valued coil combined image
            with shape (1, 320, 320) otherwise with the complex valued coil images with shape (n_coils, 320, 320).
            `None` means no augmentation.
        allowed_n_coils
            List of allowed number of coils. If `None`, all coils are allowed.
            The knee training set has 15 coils consistently, while the brain dataset has
            roughly 1300 files with 16 coils, 1100 files with 20 coils and 800 files with 4 coils.
            Only used if `coil_combine` is `False`.
        """
        slices = []
        self._filenames = []
        for fn in Path(path).rglob('*.h5') if isinstance(path, str | Path | PathLike) else path:
            try:
                with h5py.File(fn, 'r') as file:
                    acquisition: str = file.attrs['acquisition']
                    n_slices, n_coils = file['kspace'].shape[:2]
                    header = KHeader.from_ismrmrd(
                        ismrmrd.xsd.CreateFromDocument(file['ismrmrd_header'][()].decode('utf-8')), AcqInfo()
                    )
                    if acquisition.startswith('AX'):  # brain
                        if (
                            (not coil_combine and allowed_n_coils is not None and n_coils not in allowed_n_coils)
                            or header.recon_matrix.x not in (320, 384)
                            or round(header.recon_fov.y, 2) != 0.22
                        ):
                            continue
                    elif acquisition.endswith('FBK'):  # knee
                        if (
                            (not coil_combine and allowed_n_coils is not None and n_coils not in allowed_n_coils)
                            or header.recon_matrix.x != 320
                            or round(header.recon_fov.y, 2) != 0.14
                        ):
                            continue
                    else:
                        raise ValueError(f'Unknown acquisition: {acquisition}')
                    slices.append(n_slices)
                    self._filenames.append(fn)
            except (KeyError, FileNotFoundError, OSError):
                warn(f'Invalid file: {fn}. Skipping.', stacklevel=2)
        self._accum_slices = torch.tensor(slices).cumsum(dim=0)
        self._coil_combine = coil_combine
        self.augment = augment

    def __len__(self) -> int:
        """Get length (number of slices) of the dataset."""
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
            data = torch.as_tensor(np.array(file['kspace'][slice_idx]))
            header = KHeader.from_ismrmrd(
                ismrmrd.xsd.CreateFromDocument(file['ismrmrd_header'][()].decode('utf-8')), AcqInfo()
            )
            # recon_matrix can be >320, but FOV is always constant. We crop in k-space to get the same resolution.
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
                    lambda x: inati(x.unsqueeze(1), smoothing_width=5).squeeze(1), (32, 32), dim=(-2, -1)
                )(img)
                csm /= (csm * csm.conj()).sum(0, keepdim=True).sqrt()
                img = (img * csm.conj()).sum(dim=0, keepdim=True)
            if self.augment is not None:
                img = self.augment(img, idx)
            return rearrange(img, 'coils y x -> 1 coils 1 y x')
