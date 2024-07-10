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
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange
from mrpro.data._kdata.KDataRearrangeMixin import KDataRearrangeMixin
from mrpro.data._kdata.KDataRemoveOsMixin import KDataRemoveOsMixin
from mrpro.data._kdata.KDataSelectMixin import KDataSelectMixin
from mrpro.data._kdata.KDataSplitMixin import KDataSplitMixin
from mrpro.data.acq_filters import is_image_acquisition
from mrpro.data.AcqInfo import AcqInfo
from mrpro.data.EncodingLimits import Limits
from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.KTrajectoryRawShape import KTrajectoryRawShape
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd
from mrpro.utils import modify_acq_info

KDIM_SORT_LABELS = ('k1', 'k2', 'average', 'slice', 'contrast', 'phase', 'repetition', 'set')
# TODO: Consider adding the users labels here, but remember issue #32 and NOT add user5 and user6.
OTHER_LABELS = ('average', 'slice', 'contrast', 'phase', 'repetition', 'set')


@dataclasses.dataclass(slots=True, frozen=True)
class KData(KDataSplitMixin, KDataRearrangeMixin, KDataSelectMixin, KDataRemoveOsMixin, MoveDataMixin):
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
        acquisition_filter_criterion: Callable = is_image_acquisition,
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
            index of the ISMRMRD dataset to load (converter creates dataset, dataset_1, ...)
        acquisition_filter_criterion
            function which returns True if an acquisition should be included in KData
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

        acquisitions = [acq for acq in acquisitions if acquisition_filter_criterion(acq)]
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
                f'Found {n_acqs_per_other.tolist()}. \n'
                'The data will be reshaped to (acquisitions, 1, 1, k0). \n'
                'This needs to be adjusted be reshaping for a successful reconstruction. \n'
                'If unintenional, this might be caused by wrong labels in the ismrmrd file or a wrong flag filter.',
                stacklevel=1,
            )
            n_k1 = 1
            n_k2 = 1

        # Second, determine the sorting order
        acq_indices = np.stack([getattr(kheader.acq_info.idx, label) for label in KDIM_SORT_LABELS], axis=0)
        sort_idx = np.lexsort(acq_indices)  # torch does not have lexsort as of pytorch 2.2 (March 2024)

        # Finally, reshape and sort the tensors in acqinfo and acqinfo.idx, and kdata.
        def sort_and_reshape_tensor_fields(input_tensor: torch.Tensor):
            return rearrange(input_tensor[sort_idx], '(other k2 k1) ... -> other k2 k1 ...', k1=n_k1, k2=n_k2)

        kheader.acq_info = modify_acq_info(sort_and_reshape_tensor_fields, kheader.acq_info)
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

    def __repr__(self):
        """Representation method for KData class."""
        return f'KData with shape {list(self.data.shape)!s} and \n{self.header}'


class _KDataProtocol(Protocol):
    """Protocol for KData used for type hinting in KData mixins.

    Note that the actual KData class can have more properties and methods than those defined here.

    If you want to use a property or method of KData in a new KDataMixin class,
    you must add it to this Protocol to make sure that the type hinting works.

    For more information about Protocols see:
    https://typing.readthedocs.io/en/latest/spec/protocol.html#protocols
    """

    @property
    def header(self) -> KHeader: ...

    @property
    def data(self) -> torch.Tensor: ...

    @property
    def traj(self) -> KTrajectory: ...

    def __init__(self, header: KHeader, data: torch.Tensor, traj: KTrajectory): ...

    def _split_k2_or_k1_into_other(
        self, split_idx: torch.Tensor, other_label: str, split_dir: str
    ) -> _KDataProtocol: ...
