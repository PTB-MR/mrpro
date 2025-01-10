"""MR raw data / k-space data class."""

import copy
import dataclasses
import datetime
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from types import EllipsisType
from typing import Literal, cast

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange, repeat
from typing_extensions import Self, TypeVar

from mrpro.data.acq_filters import has_n_coils, is_image_acquisition
from mrpro.data.AcqInfo import AcqInfo, rearrange_acq_info_fields
from mrpro.data.EncodingLimits import Limits
from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.KTrajectoryRawShape import KTrajectoryRawShape
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd

RotationOrTensor = TypeVar('RotationOrTensor', bound=torch.Tensor | Rotation)

KDIM_SORT_LABELS = (
    'k1',
    'k2',
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user7',
)

OTHER_LABELS = (
    'average',
    'slice',
    'contrast',
    'phase',
    'repetition',
    'set',
    'user0',
    'user1',
    'user2',
    'user3',
    'user4',
    'user7',
)


@dataclasses.dataclass(slots=True, frozen=True)
class KData(
    MoveDataMixin,
):
    """MR raw data / k-space data class."""

    header: KHeader
    """Header information for k-space data"""

    data: torch.Tensor
    """K-space data. Shape (...other coils k2 k1 k0)"""

    traj: KTrajectory
    """K-space trajectory along kz, ky and kx. Shape (...other k2 k1 k0)"""

    @classmethod
    def from_file(
        cls,
        filename: str | Path,
        ktrajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: dict[str, object] | None = None,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable = is_image_acquisition,
    ) -> Self:
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

        # we need the same number of receiver coils for all acquisitions
        n_coils_available = {acq.data.shape[0] for acq in acquisitions}
        if len(n_coils_available) > 1:
            if (
                ismrmrd_header.acquisitionSystemInformation is not None
                and ismrmrd_header.acquisitionSystemInformation.receiverChannels is not None
            ):
                n_coils = int(ismrmrd_header.acquisitionSystemInformation.receiverChannels)
            else:
                # most likely, highest number of elements are the coils used for imaging
                n_coils = int(max(n_coils_available))

            warnings.warn(
                f'Acquisitions with different number {n_coils_available} of receiver coil elements detected. '
                f'Data with {n_coils} receiver coil elements will be used.',
                stacklevel=1,
            )
            acquisitions = [acq for acq in acquisitions if has_n_coils(n_coils, acq)]

        if not acquisitions:
            raise ValueError('No acquisitions meeting the given filter criteria were found.')

        kdata = torch.stack([torch.as_tensor(acq.data, dtype=torch.complex64) for acq in acquisitions])

        acqinfo = AcqInfo.from_ismrmrd_acquisitions(acquisitions)

        if len(torch.unique(acqinfo.idx.user5)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 5 indices for storing the kspace center line number.\n'
                'User 5 indices will be ignored',
                stacklevel=1,
            )

        if len(torch.unique(acqinfo.idx.user6)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 6 indices for storing the kspace center partition number.\n'
                'User 6 indices will be ignored',
                stacklevel=1,
            )

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
        kheader.acq_info.apply_(
            lambda field: rearrange_acq_info_fields(
                field[sort_idx], '(other k2 k1) ... -> other k2 k1 ...', k1=n_k1, k2=n_k2
            )
            if isinstance(field, torch.Tensor | Rotation)
            else field
        )
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
        traj = KTrajectory(self.traj.kz, self.traj.ky, self.traj.kx)
        try:
            device = str(self.device)
        except RuntimeError:
            device = 'mixed'
        out = (
            f'{type(self).__name__} with shape {list(self.data.shape)!s} and dtype {self.data.dtype}\n'
            f'Device: {device}\n'
            f'{traj}\n'
            f'{self.header}'
        )
        return out

    def compress_coils(
        self: Self,
        n_compressed_coils: int,
        batch_dims: None | Sequence[int] = None,
        joint_dims: Sequence[int] | EllipsisType = ...,
    ) -> Self:
        """Reduce the number of coils based on a PCA compression.

        A PCA is carried out along the coil dimension and the n_compressed_coils virtual coil elements are selected. For
        more information on coil compression please see [BUE2007]_, [DON2008]_ and [HUA2008]_.

        Returns a copy of the data.

        Parameters
        ----------
        kdata
            K-space data
        n_compressed_coils
            Number of compressed coils
        batch_dims
            Dimensions which are treated as batched, i.e. separate coil compression matrizes (e.g. different slices).
            Default is to do one coil compression matrix for the entire k-space data. Only batch_dim or joint_dim can
            be defined. If batch_dims is not None then joint_dims has to be ...
        joint_dims
            Dimensions which are combined to calculate single coil compression matrix (e.g. k0, k1, contrast). Default
            is that all dimensions (except for the coil dimension) are joint_dims. Only batch_dim or joint_dim can
            be defined. If joint_dims is not ... batch_dims has to be None

        Returns
        -------
            Copy of K-space data with compressed coils.

        Raises
        ------
        ValueError
            If both batch_dims and joint_dims are defined.
        Valuer Error
            If coil dimension is part of joint_dims or batch_dims.

        References
        ----------
        .. [BUE2007] Buehrer M, Pruessmann KP, Boesiger P, Kozerke S (2007) Array compression for MRI with large coil
           arrays. MRM 57. https://doi.org/10.1002/mrm.21237
        .. [DON2008] Doneva M, Boernert P (2008) Automatic coil selection for channel reduction in SENSE-based parallel
           imaging. MAGMA 21. https://doi.org/10.1007/s10334-008-0110-x
        .. [HUA2008] Huang F, Vijayakumar S, Li Y, Hertel S, Duensing GR (2008) A software channel compression
           technique for faster reconstruction with many channels. MRM 26. https://doi.org/10.1016/j.mri.2007.04.010

        """
        from mrpro.operators import PCACompressionOp

        coil_dim = -4 % self.data.ndim

        if n_compressed_coils > (n_current_coils := self.data.shape[coil_dim]):
            raise ValueError(
                f'Number of compressed coils ({n_compressed_coils}) cannot be greater '
                f'than the number of current coils ({n_current_coils}).'
            )

        if batch_dims is not None and joint_dims is not Ellipsis:
            raise ValueError('Either batch_dims or joint_dims can be defined not both.')

        if joint_dims is not Ellipsis:
            joint_dims_normalized = [i % self.data.ndim for i in joint_dims]
            if coil_dim in joint_dims_normalized:
                raise ValueError('Coil dimension must not be in joint_dims')
            batch_dims_normalized = [
                d for d in range(self.data.ndim) if d not in joint_dims_normalized and d is not coil_dim
            ]
        else:
            batch_dims_normalized = [] if batch_dims is None else [i % self.data.ndim for i in batch_dims]
            if coil_dim in batch_dims_normalized:
                raise ValueError('Coil dimension must not be in batch_dims')

        # reshape to (*batch dimension, -1, coils)
        permute_order = (
            *batch_dims_normalized,
            *[i for i in range(self.data.ndim) if i != coil_dim and i not in batch_dims_normalized],
            coil_dim,
        )
        kdata_permuted = self.data.permute(permute_order)
        kdata_flattened = kdata_permuted.flatten(
            start_dim=len(batch_dims_normalized), end_dim=-2
        )  # keep separate dimensions and coil

        pca_compression_op = PCACompressionOp(data=kdata_flattened, n_components=n_compressed_coils)
        (kdata_coil_compressed_flattened,) = pca_compression_op(kdata_flattened)
        del kdata_flattened
        # reshape to original dimensions and undo permutation
        kdata_coil_compressed = torch.reshape(
            kdata_coil_compressed_flattened, [*kdata_permuted.shape[:-1], n_compressed_coils]
        ).permute(*np.argsort(permute_order))

        return type(self)(self.header.clone(), kdata_coil_compressed, self.traj.clone())

    def rearrange_k2_k1_into_k1(self: Self) -> Self:
        """Rearrange kdata from (... k2 k1 ...) to (... 1 (k2 k1) ...).

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)

        Returns
        -------
            K-space data (other coils 1 (k2 k1) k0)
        """
        # Rearrange data
        kdat = rearrange(self.data, '... coils k2 k1 k0->... coils 1 (k2 k1) k0')

        # Rearrange trajectory
        ktraj = rearrange(self.traj.as_tensor(), 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0')

        # Create new header with correct shape
        kheader = copy.deepcopy(self.header)

        # Update shape of acquisition info index
        kheader.acq_info.apply_(
            lambda field: rearrange_acq_info_fields(field, 'other k2 k1 ... -> other 1 (k2 k1) ...')
        )

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))

    def remove_readout_os(self: Self) -> Self:
        """Remove any oversampling along the readout (k0) direction [GAD]_.

        Returns a copy of the data.

        Parameters
        ----------
        kdata
            K-space data

        Returns
        -------
            Copy of K-space data with oversampling removed.

        Raises
        ------
        ValueError
            If the recon matrix along x is larger than the encoding matrix along x.

        References
        ----------
        .. [GAD] Gadgetron https://github.com/gadgetron/gadgetron-python
        """
        from mrpro.operators.FastFourierOp import FastFourierOp

        # Ratio of k0/x between encoded and recon space
        x_ratio = self.header.recon_matrix.x / self.header.encoding_matrix.x
        if x_ratio == 1:
            # If the encoded and recon space is the same we don't have to do anything
            return self
        elif x_ratio > 1:
            raise ValueError('Recon matrix along x should be equal or larger than encoding matrix along x.')

        # Starting and end point of image after removing oversampling
        start_cropped_readout = (self.header.encoding_matrix.x - self.header.recon_matrix.x) // 2
        end_cropped_readout = start_cropped_readout + self.header.recon_matrix.x

        def crop_readout(data_to_crop: torch.Tensor) -> torch.Tensor:
            # returns a cropped copy
            return data_to_crop[..., start_cropped_readout:end_cropped_readout].clone()

        # Transform to image space along readout, crop to reconstruction matrix size and transform back
        fourier_k0_op = FastFourierOp(dim=(-1,))
        (cropped_data,) = fourier_k0_op(crop_readout(*fourier_k0_op.H(self.data)))

        # Adapt trajectory
        ks = [self.traj.kz, self.traj.ky, self.traj.kx]
        # only cropped ks that are not broadcasted/singleton along k0
        cropped_ks = [crop_readout(k) if k.shape[-1] > 1 else k.clone() for k in ks]
        cropped_traj = KTrajectory(cropped_ks[0], cropped_ks[1], cropped_ks[2])

        # Adapt header parameters
        header = copy.deepcopy(self.header)
        header.acq_info.center_sample -= start_cropped_readout
        header.acq_info.number_of_samples[:] = cropped_data.shape[-1]
        header.encoding_matrix.x = cropped_data.shape[-1]

        header.acq_info.discard_post = (header.acq_info.discard_post * x_ratio).to(torch.int32)
        header.acq_info.discard_pre = (header.acq_info.discard_pre * x_ratio).to(torch.int32)

        return type(self)(header, cropped_data, cropped_traj)

    def select_other_subset(
        self: Self,
        subset_idx: torch.Tensor,
        subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Select a subset from the other dimension of KData.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        subset_idx
            Index which elements of the other subset to use, e.g. phase 0,1,2 and 5
        subset_label
            Name of the other label, e.g. phase

        Returns
        -------
            K-space data (other_subset coils k2 k1 k0)

        Raises
        ------
        ValueError
            If the subset indices are not available in the data
        """
        # Make a copy such that the original kdata.header remains the same
        kheader = copy.deepcopy(self.header)
        ktraj = self.traj.as_tensor()

        # Verify that the subset_idx is available
        label_idx = getattr(kheader.acq_info.idx, subset_label)
        if not all(el in torch.unique(label_idx) for el in subset_idx):
            raise ValueError('Subset indices are outside of the available index range')

        # Find subset index in acq_info index
        other_idx = torch.cat([torch.where(idx == label_idx[:, 0, 0])[0] for idx in subset_idx], dim=0)

        # Adapt header
        kheader.acq_info.apply_(
            lambda field: field[other_idx, ...] if isinstance(field, torch.Tensor | Rotation) else field
        )

        # Select data
        kdat = self.data[other_idx, ...]

        # Select ktraj
        if ktraj.shape[1] > 1:
            ktraj = ktraj[:, other_idx, ...]

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))

    def _split_k2_or_k1_into_other(
        self,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
        split_dir: Literal['k2', 'k1'],
    ) -> Self:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        split_idx
            2D index describing the k2 or k1 points in each block to be moved to the other dimension
            (other_split, k1_per_split) or (other_split, k2_per_split)
        other_label
            Label of other dimension, e.g. repetition, phase
        split_dir
            Dimension to split, either 'k1' or 'k2'

        Returns
        -------
            K-space data with new shape
            ((other other_split) coils k2 k1_per_split k0) or ((other other_split) coils k2_per_split k1 k0)

        Raises
        ------
        ValueError
            Already existing "other_label" can only be of length 1
        """
        # Number of other
        n_other = split_idx.shape[0]

        # Verify that the specified label of the other dimension is unused
        if getattr(self.header.encoding_limits, other_label).length > 1:
            raise ValueError(f'{other_label} is already used to encode different parts of the scan.')

        # Set-up splitting
        if split_dir == 'k1':
            # Split along k1 dimensions
            def split_data_traj(dat_traj: torch.Tensor) -> torch.Tensor:
                return dat_traj[:, :, :, split_idx, :]

            def split_acq_info(acq_info: RotationOrTensor) -> RotationOrTensor:
                # cast due to https://github.com/python/mypy/issues/10817
                return cast(RotationOrTensor, acq_info[:, :, split_idx, ...])

            # Rearrange other_split and k1 dimension
            rearrange_pattern_data = 'other coils k2 other_split k1 k0->(other other_split) coils k2 k1 k0'
            rearrange_pattern_traj = 'dim other k2 other_split k1 k0->dim (other other_split) k2 k1 k0'
            rearrange_pattern_acq_info = 'other k2 other_split k1 ... -> (other other_split) k2 k1 ...'

        elif split_dir == 'k2':
            # Split along k2 dimensions
            def split_data_traj(dat_traj: torch.Tensor) -> torch.Tensor:
                return dat_traj[:, :, split_idx, :, :]

            def split_acq_info(acq_info: RotationOrTensor) -> RotationOrTensor:
                return cast(RotationOrTensor, acq_info[:, split_idx, ...])

            # Rearrange other_split and k1 dimension
            rearrange_pattern_data = 'other coils other_split k2 k1 k0->(other other_split) coils k2 k1 k0'
            rearrange_pattern_traj = 'dim other other_split k2 k1 k0->dim (other other_split) k2 k1 k0'
            rearrange_pattern_acq_info = 'other other_split k2 k1 ... -> (other other_split) k2 k1 ...'

        else:
            raise ValueError('split_dir has to be "k1" or "k2"')

        # Split data
        kdat = rearrange(split_data_traj(self.data), rearrange_pattern_data)

        # First we need to make sure the other dimension is the same as data then we can split the trajectory
        ktraj = self.traj.as_tensor()
        # Verify that other dimension of trajectory is 1 or matches data
        if ktraj.shape[1] > 1 and ktraj.shape[1] != self.data.shape[0]:
            raise ValueError(f'other dimension of trajectory has to be 1 or match data ({self.data.shape[0]})')
        elif ktraj.shape[1] == 1 and self.data.shape[0] > 1:
            ktraj = repeat(ktraj, 'dim other k2 k1 k0->dim (other_data other) k2 k1 k0', other_data=self.data.shape[0])
        ktraj = rearrange(split_data_traj(ktraj), rearrange_pattern_traj)

        # Create new header with correct shape
        kheader = self.header.clone()

        # Update shape of acquisition info index
        kheader.acq_info.apply_(
            lambda field: rearrange_acq_info_fields(split_acq_info(field), rearrange_pattern_acq_info)
            if isinstance(field, Rotation | torch.Tensor)
            else field
        )

        # Update other label limits and acquisition info
        setattr(kheader.encoding_limits, other_label, Limits(min=0, max=n_other - 1, center=0))

        # acq_info for new other dimensions
        acq_info_other_split = repeat(
            torch.linspace(0, n_other - 1, n_other), 'other-> other k2 k1', k2=kdat.shape[-3], k1=kdat.shape[-2]
        )
        setattr(kheader.acq_info.idx, other_label, acq_info_other_split)

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))

    def split_k1_into_other(
        self: Self,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        split_idx
            2D index describing the k1 points in each block to be moved to other dimension  (other_split, k1_per_split)
        other_label
            Label of other dimension, e.g. repetition, phase

        Returns
        -------
            K-space data with new shape ((other other_split) coils k2 k1_per_split k0)
        """
        return self._split_k2_or_k1_into_other(split_idx, other_label, split_dir='k1')

    def split_k2_into_other(
        self: Self,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        split_idx
            2D index describing the k2 points in each block to be moved to other dimension  (other_split, k2_per_split)
        other_label
            Label of other dimension, e.g. repetition, phase

        Returns
        -------
            K-space data with new shape ((other other_split) coils k2_per_split k1 k0)
        """
        return self._split_k2_or_k1_into_other(split_idx, other_label, split_dir='k2')
