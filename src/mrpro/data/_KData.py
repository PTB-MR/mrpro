"""MR raw data / k-space data class."""

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
import datetime
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Self
from typing import TypeAlias

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange

from mrpro.data import AcqIdx
from mrpro.data import AcqInfo
from mrpro.data import Data
from mrpro.data import KHeader
from mrpro.data import KTrajectory
from mrpro.data import KTrajectoryRawShape
from mrpro.data import Limits
from mrpro.data._AcqInfo import AcqIdxLiteral
from mrpro.data.enums import AcqFlags
from mrpro.data.traj_calculators import KTrajectoryCalculator
from mrpro.data.traj_calculators import KTrajectoryIsmrmrd

ItemType: TypeAlias = torch.Tensor

KDIM_SORT_LABELS = ('k1', 'k2', 'average', 'slice', 'contrast', 'phase', 'repetition', 'set')
# TODO: Consider adding the users labels here, but remember issue #32 and NOT add user5 and user6.
OTHER_LABELS = ('average', 'slice', 'contrast', 'phase', 'repetition', 'set')

# Same criteria as https://github.com/wtclarke/pymapvbvd/blob/master/mapvbvd/mapVBVD.py uses
DEFAULT_IGNORE_FLAGS = (
    AcqFlags.ACQ_IS_NOISE_MEASUREMENT
    | AcqFlags.ACQ_IS_DUMMYSCAN_DATA
    | AcqFlags.ACQ_IS_HPFEEDBACK_DATA
    | AcqFlags.ACQ_IS_NAVIGATION_DATA
    | AcqFlags.ACQ_IS_PHASECORR_DATA
    | AcqFlags.ACQ_IS_PHASE_STABILIZATION
    | AcqFlags.ACQ_IS_PHASE_STABILIZATION_REFERENCE
    | AcqFlags.ACQ_IS_PARALLEL_CALIBRATION
)


@dataclasses.dataclass(slots=True, frozen=True)
class KData(Data):
    """MR raw data / k-space data class."""

    header: KHeader
    data: torch.Tensor
    traj: KTrajectory

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        ktrajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: dict[str, object] | None = None,
        dataset_idx: int = -1,
        ignore_flags: AcqFlags = DEFAULT_IGNORE_FLAGS,
    ) -> KData:
        """Load k-space data from an ISMRMRD file.

        Parameters
        ----------
        filename
            path to the ISMRMRD file
        ktrajectory
            KTrajectoryCalculator to calculate the k-space trajectory or an already calculated KTrajectory
        header_overwrites
            dictionary of key-value pairs to overwrite the header
        dataset_idx
            index of the ISMRMRD dataset to load (converter creates dataset, dataset_1, ...), default is -1 (last)
        ignore_flags
            Acqisition flags to filter out. Defaults to all non-images as defined by pymapvbvd.
            Use ACQ_NO_FLAG to disable the filter.
            Note: If ACQ_IS_PARALLEL_CALIBRATION is set without also setting ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
                    we interpret it as: Ignore if IS_PARALLEL_CALIBRATION and not PARALLEL_CALIBRATION_AND_IMAGING
        """
        # Can raise FileNotFoundError
        with ismrmrd.File(filename, 'r') as file:
            dataset = file[list(file.keys())[dataset_idx]]
            ismrmrd_header = dataset.header
            acquisitions = dataset.acquisitions[:]
            try:
                mtime: int = h5py.h5g.get_objinfo(dataset['data']._contents.id).mtime
            except AttributeError:
                mtime = 0
            modification_time = datetime.datetime.fromtimestamp(mtime)

        if (
            AcqFlags.ACQ_IS_PARALLEL_CALIBRATION in ignore_flags
            and AcqFlags.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING not in ignore_flags
        ):
            # if only ACQ_IS_PARALLEL_CALIBRATION is set, reinterpret it as: ignore if
            # ACQ_IS_PARALLEL_CALIBRATION is set and ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING is not set
            ignore_flags = ignore_flags & ~AcqFlags.ACQ_IS_PARALLEL_CALIBRATION
            acquisitions = list(
                filter(
                    lambda acq: (AcqFlags.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING.value & acq.flags)
                    or not (AcqFlags.ACQ_IS_PARALLEL_CALIBRATION.value & acq.flags),
                    acquisitions,
                ),
            )

        acquisitions = list(filter(lambda acq: not (ignore_flags.value & acq.flags), acquisitions))
        kdata = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        acqinfo = AcqInfo.from_ismrmrd_acquisitions(acquisitions)

        # Raises ValueError if required fields are missing in the header
        kheader = KHeader.from_ismrmrd(
            ismrmrd_header,
            acqinfo,
            defaults={
                'datetime': modification_time,  # use the modification time of the dataset as fallback
                'trajectory': ktrajectory,
            },
            overwrite=header_overwrites,
        )

        # Fill k0 limits if they were set to zero / not set in the header
        if kheader.encoding_limits.k0.length == 1:
            # The encodig limits should describe the encoded space of all readout lines. If we have two readouts with
            # (number_of_samples, center_sample) of (100, 20) (e.g. partial Fourier in the negative k0 direction) and
            # (100, 80) (e.g. partial Fourier in the positive k0 direction) then this should lead to encoding limits of
            # [min=0, max=159, center=80]
            max_center_sample = int(torch.max(kheader.acq_info.center_sample))
            max_pos_k0_extend = int(torch.max(kheader.acq_info.number_of_samples - kheader.acq_info.center_sample))
            kheader.encoding_limits.k0 = Limits(0, max_center_sample + max_pos_k0_extend - 1, max_center_sample)

        # Sort and reshape the kdata and the acquisistion info according to the indices.
        # within "other", the aquisistions are sorted in the order determined by KDIM_SORT_LABELS.
        # The final shape will be ("all other labels", k2, k1, k0) for kdata
        # and ("all other labels", k2, k1, length of the aqusitions info field) for aquisistion info.

        # First, determine if we can split into k2 and k1 and how large these should be
        acq_indices_other = torch.stack([getattr(kheader.acq_info.idx, label) for label in OTHER_LABELS], dim=0)
        _, n_acqs_per_other = torch.unique(acq_indices_other, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of the label values in "other"
        n_acqs_per_other = torch.unique(n_acqs_per_other)

        acq_indices_other_k2 = torch.cat((acq_indices_other, kheader.acq_info.idx.k2.unsqueeze(0)), dim=0)
        _, n_acqs_per_other_and_k2 = torch.unique(acq_indices_other_k2, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of other **and k2**
        n_acqs_per_other_and_k2 = torch.unique(n_acqs_per_other_and_k2)

        if len(n_acqs_per_other_and_k2) == 1:
            # This is the most common case:
            # All other and k2 combinations have the same number of aquisistions which we can use as k1
            # to reshape to (other, k2, k1, k0)
            n_k1 = n_acqs_per_other_and_k2[0]
            n_k2 = n_acqs_per_other[0] // n_k1
        elif len(n_acqs_per_other) == 1:
            # We cannot split the data along phase encoding steps into k2 and k1, as there are different numbers of k1.
            # But all "other labels" combinations have the same number of aquisisitions,
            # so we can reshape to (other, k, 1, k0)
            n_k1 = 1
            n_k2 = n_acqs_per_other[0]  # total number of k encoding steps
        else:
            # For different other combinations we have different numbers of aquisistions,
            # so we can only reshape to (acquisitions, 1, 1, k0)
            # This might be an user error.
            warnings.warn(
                f'There are different numbers of acquisistions in'
                'different combinations of labels {"/".join(OTHER_LABELS)}: \n'
                f'Found {n_acqs_per_other.tolist()}.'
                'The data will be reshaped to (acquisitions, 1, 1, k0). \n'
                'If unintenional, this might be caused by wrong labels in the ismrmrd file or a wrong flag filter.',
                stacklevel=1,
            )
            n_k1 = 1
            n_k2 = 1

        # Second, determine the sorting order
        acq_indices = np.stack([getattr(kheader.acq_info.idx, label) for label in KDIM_SORT_LABELS], axis=0)
        sort_idx = np.lexsort(acq_indices)  # torch does not have lexsort as of pytorch 2.2 (March 2024)

        # Finally, reshape and sort the tensors in acqinfo and acqinfo.idx, and kdata.
        def sort_and_reshape_tensor_fields(dataclass: AcqInfo | AcqIdx):
            """Sort by the sort_idx and reshape to (*, n_k2, n_k1, ...)."""
            for field in dataclasses.fields(dataclass):
                old = getattr(dataclass, field.name)
                if isinstance(old, torch.Tensor):
                    new = rearrange(old[sort_idx], '(other k2 k1) ... -> other k2 k1 ...', k1=n_k1, k2=n_k2)
                    setattr(dataclass, field.name, new)

        sort_and_reshape_tensor_fields(kheader.acq_info)
        sort_and_reshape_tensor_fields(kheader.acq_info.idx)

        kdata = rearrange(kdata[sort_idx], '(other k2 k1) coils k0 -> other coils k2 k1 k0', k1=n_k1, k2=n_k2)

        # Calculate trajectory and check if it matches the kdata shape
        match ktrajectory:
            case KTrajectoryIsmrmrd():
                ktrajectory_final = ktrajectory(acquisitions).sort_and_reshape(sort_idx, n_k2, n_k1)
            case KTrajectoryCalculator():
                ktrajectory_or_rawshape = ktrajectory(kheader)
                if isinstance(ktrajectory_or_rawshape, KTrajectoryRawShape):
                    ktrajectory_final = ktrajectory_or_rawshape.sort_and_reshape(sort_idx, n_k2, n_k1)
                else:
                    ktrajectory_final = ktrajectory_or_rawshape
            case KTrajectory():
                ktrajectory_final = ktrajectory
            case _:
                raise TypeError(
                    'ktrajectory must be KTrajectoryIsmrmrd, KTrajectory or KTrajectoryCalculator'
                    f'not {type(ktrajectory)}',
                )

        try:
            shape = ktrajectory_final.broadcasted_shape
            torch.broadcast_shapes(kdata[..., 0, :, :, :].shape, shape)
        except RuntimeError:
            # Not broadcastable
            raise ValueError(
                f'Broadcasted shape trajectory do not match kdata: {shape} vs. {kdata.shape}. '
                'Please check the trajectory.',
            ) from None

        return cls(kheader, kdata, ktrajectory_final)

    def clone(self):
        return KData(
            header=self.header.clone(),
            data=self.data.clone(),
            traj=self.traj.clone(),
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: None | torch.dtype = None,
        non_blocking: bool = False,
        copy: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> KData:
        """Perform dtype and/or device conversion of trajectory and data.

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
        # Only complex64 and complex128 supported for kdata.
        # This will then lead to a trajectory of float32 and float64, respectively.
        if dtype is None:
            dtype_traj = None
        elif dtype == torch.complex64:
            dtype_traj = torch.float32
        elif dtype == torch.complex128:
            dtype_traj = torch.float64
        else:
            raise ValueError(f'dtype {dtype} not supported. Only torch.complex64 and torch.complex128 is supported.')

        return KData(
            header=self.header,
            data=self.data.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            ),
            traj=self.traj.to(
                device=device,
                dtype=dtype_traj,
                non_blocking=non_blocking,
                copy=copy,
                memory_format=memory_format,
            ),
        )

    def cuda(
        self,
        device: torch.device | None = None,
        non_blocking: bool = False,
        memory_format: torch.memory_format = torch.preserve_format,
    ) -> KData:
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
        return KData(
            header=self.header,
            data=self.data.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format),  # type: ignore [call-arg]
            traj=self.traj.cuda(device=device, non_blocking=non_blocking, memory_format=memory_format),
        )

    def cpu(self, memory_format: torch.memory_format = torch.preserve_format) -> KData:
        """Create copy of object in CPU memory.

        Parameters
        ----------
        memory_format
            The desired memory format of returned Tensor.
        """
        return KData(
            header=self.header,
            data=self.data.cpu(memory_format=memory_format),  # type: ignore [call-arg]
            traj=self.traj.cpu(memory_format=memory_format),
        )

    def __getitem__(self, item: ItemType) -> KDataView:
        traj = self.traj[item]
        header = self.header[item]
        data = self.data[item]
        return KDataView(header, data, traj, self)

    def rearrange(self, pattern: str, **axes_length) -> Self:
        """Rearrange the data, trajectory, and indices according to the pattern.

        The pattern is based on einops.rearrange notation, with the extension of '|'
        signifying the coil dimensio in the data and the delimiter between k-dimensions and the other dimensions.

        Examples
        --------
        Given a data shape of (other, coils, k2, k1, k0)
        - The pattern '(other1 other2) | k2 k1 k0 -> other1 other2 | k2 k1 k0 other'
            with additional arguments other2=2
            would split the other dimension into two dimensions, resulting in the shape
            (other//2, 2, coils, k2, k1, k0)
          This can also be written as
          - '(other1 other2) | ... -> other1 other2 | ...' using ellipsis to indicate all other dimensions
          - '(other1 other2) | ... -> other1 other2'
             using an implicit '| ...' at the end if no '|' is in the pattern.
        - The pattern ... | ... -> (...) | (...)
          would result in the shape (other, coils, 1, 1, k2*k1*k2)
          All other dimensions are flattened. All k-dimensions are flattened.
          The existence at least one 'other' dimension, the coil dimension and k2, k1, and k0
          is always enforced by introducing singleton dimensions after the rearrange.

        Parameters
        ----------
        pattern
            Rearrangement pattern based on einops notation. See above for some examples.
        **axes_length
            any additional specifications for dimensions

        Returns
        -------
            A new KData object with the rearranged data, trajectory, and indices.
            All information is cloned.
        """
        trajectory = self.traj.rearrange(pattern, **axes_length)
        header = self.header.clone()
        header.acq_info = header.acq_info.rearrange(pattern, **axes_length)

        # rearrange the data
        raise NotImplementedError('rearrange data')

    def select_idx_subset(
        self,
        subset_idx: torch.Tensor | Sequence[int],
        subset_label: AcqIdxLiteral,
    ) -> Self:
        """Select a subset KData based on the AcqIdx.

        Parameters
        ----------
        subset_idx
            Index which elements of the subset to use, e.g. to use phase 0,1,2 and 5
            supply torch.tensor([0,1,2,5])
        subset_label
            Name of the subset, e.g. phase

        Returns
        -------
            A new KData object (copy) with the selected subset
        """
        subset_idx_tensor = torch.as_tensor(subset_idx)
        idx = getattr(self.header.acq_info.idx, subset_label)
        subset_idx_tensor = subset_idx_tensor.to(idx.device)

        idx_does_exist = torch.isin(subset_idx_tensor, torch.unique(idx))
        if not torch.all(idx_does_exist):
            missing = subset_idx_tensor[~idx_does_exist]
            raise ValueError(f'Subset indices {missing} are not in the available indices.')

        mask = torch.isin(idx, subset_idx_tensor)
        header = self.header.clone()
        header.acq_info = self.header.acq_info[mask]
        trajectory = self.traj[mask].clone()
        data = self.data[mask].clone()
        return type(self)(header, data, trajectory)

    def arange_idx_(self, dim: int, idx_label: AcqIdxLiteral):
        """Overwrite the index attribute in the acquisition based on the data shape.

        The index attribute is overwritten with a range from 0 to the length of the data in the specified dimension.
        This can be used to assign a new index to the data, e.g. after rearranging the data.

        Parameters
        ----------
        dim
            Dimension of the data to consider.
            The target dimensions in the AcqIdx will the corresponding dimension.
            Cannot be the coil dimension or the k0 dimension.
        idx_label
            The name of the AcqIdx attribute to overwrite as a string.

        """
        shape = self.data.shape
        if dim >= len(shape) or dim < -len(shape):
            raise ValueError(f'dim {dim} is out of range for data with shape {shape}')
        dim = dim % len(shape)
        if dim == len(shape) - 1:
            raise ValueError('Cannot use the k0 dimension for arange_idx_, as it is not encoded in the acquisition idx')
        if dim == len(shape) - 4:
            raise ValueError(
                'Cannot use the coil dimension for arange_idx_, as it is not encoded in the acquisition idx'
            )

        idx_old = getattr(self.header.acq_info.idx, idx_label)
        idx_new = torch.arange(shape[dim], device=idx_old.device, dtype=idx_old.dtype)
        target_dim = dim - 1 if (dim > len(shape) - 4) else dim  # skip the coil dimension
        idx_new = idx_new[..., (None,) * (idx_old.ndim - target_dim - 1)]
        idx_new.expand(*idx_old.shape[:target_dim], -1, *idx_old.shape[target_dim + 1 :])
        setattr(self.header.acq_info.idx, idx_label, idx_new)

    # def _split_k2_or_k1_into_other(
    #         self,
    #         split_idx: torch.Tensor,
    #         other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    #         split_dir: Literal['k2', 'k1'],
    #     ) -> KData:
    #         """Based on an index tensor, split the data in e.g. phases.

    #         Parameters
    #         ----------
    #         split_idx
    #             2D index describing the k2 or k1 points in each block to be moved to the other dimension
    #             (other_split, k1_per_split) or (other_split, k2_per_split)
    #         other_label
    #             Label of other dimension, e.g. repetition, phase
    #         split_dir
    #             Dimension to split, either 'k1' or 'k2'

    #         Returns
    #         -------
    #             K-space data with new shape
    #             ((other other_split) coils k2 k1_per_split k0) or ((other other_split) coils k2_per_split k1 k0)

    #         Raises
    #         ------
    #         ValueError
    #             Already existing "other_label" can only be of length 1
    #         """
    #         # Number of other
    #         n_other = split_idx.shape[0]

    #         # Verify that the specified label of the other dimension is unused
    #         if getattr(self.header.encoding_limits, other_label).length > 1:
    #             raise ValueError(f'{other_label} is already used to encode different parts of the scan.')

    #         # Set-up splitting
    #         if split_dir == 'k1':
    #             # Split along k1 dimensions
    #             def split_data_traj(dat_traj: torch.Tensor) -> torch.Tensor:
    #                 return dat_traj[:, :, :, split_idx, :]

    #             def split_acq_info(acq_info: torch.Tensor) -> torch.Tensor:
    #                 return acq_info[:, :, split_idx, ...]

    #             # Rearrange other_split and k1 dimension
    #             rearrange_pattern_data = 'other coils k2 other_split k1 k0->(other other_split) coils k2 k1 k0'
    #             rearrange_pattern_traj = 'dim other k2 other_split k1 k0->dim (other other_split) k2 k1 k0'
    #             rearrange_pattern_acq_info = 'other k2 other_split k1 ... -> (other other_split) k2 k1 ...'

    #         elif split_dir == 'k2':
    #             # Split along k2 dimensions
    #             def split_data_traj(dat_traj: torch.Tensor) -> torch.Tensor:
    #                 return dat_traj[:, :, split_idx, :, :]

    #             def split_acq_info(acq_info: torch.Tensor) -> torch.Tensor:
    #                 return acq_info[:, split_idx, ...]

    #             # Rearrange other_split and k1 dimension
    #             rearrange_pattern_data = 'other coils other_split k2 k1 k0->(other other_split) coils k2 k1 k0'
    #             rearrange_pattern_traj = 'dim other other_split k2 k1 k0->dim (other other_split) k2 k1 k0'
    #             rearrange_pattern_acq_info = 'other other_split k2 k1 ... -> (other other_split) k2 k1 ...'

    #         else:
    #             raise ValueError('split_dir has to be "k1" or "k2"')

    #         # Split data
    #         kdat = rearrange(split_data_traj(self.data), rearrange_pattern_data)

    #         # First we need to make sure the other dimension is the same as data then we can split the trajectory
    #         ktraj = self.traj.as_tensor()
    #         # Verify that other dimension of trajectory is 1 or matches data
    #         if ktraj.shape[1] > 1 and ktraj.shape[1] != self.data.shape[0]:
    #             raise ValueError(f'other dimension of trajectory has to be 1 or match data ({self.data.shape[0]})')
    #         elif ktraj.shape[1] == 1 and self.data.shape[0] > 1:
    #             ktraj = repeat(ktraj, 'dim other k2 k1 k0->dim (other_data other) k2 k1 k0', other_data=self.data.shape[0])
    #         ktraj = rearrange(split_data_traj(ktraj), rearrange_pattern_traj)

    #         # Create new header with correct shape
    #         kheader = copy.deepcopy(self.header)

    #         # Update shape of acquisition info index
    #         def reshape_acq_info(info: torch.Tensor):
    #             return rearrange(split_acq_info(info), rearrange_pattern_acq_info)

    #         kheader.acq_info = modify_acq_info(reshape_acq_info, kheader.acq_info)

    #         # Update other label limits and acquisition info
    #         setattr(kheader.encoding_limits, other_label, Limits(min=0, max=n_other - 1, center=0))

    #         # acq_info for new other dimensions
    #         acq_info_other_split = repeat(
    #             torch.linspace(0, n_other - 1, n_other), 'other-> other k2 k1', k2=kdat.shape[-3], k1=kdat.shape[-2]
    #         )
    #         setattr(kheader.acq_info.idx, other_label, acq_info_other_split)

    #         return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))


class KDataView(KData):
    """View of KData.

    This class is used to create a view of a KData object. The data is not copied, but shared with the original KData.
    """

    _base: KData

    def __init__(self, header, data, traj, base: KData):
        """Create a new KDataView object.

        Parameters
        ----------
        kdata
            KData object to create a view of
        item
            Index of the view
        """
        self.header = header
        self.data = data
        self.traj = traj
        self._base = base
