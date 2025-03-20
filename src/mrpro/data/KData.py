"""MR raw data / k-space data class."""

import copy
import dataclasses
import datetime
import warnings
from collections.abc import Callable, Sequence
from types import EllipsisType
from typing import Literal, cast

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange, repeat
from typing_extensions import Self, TypeVar

from mrpro.data.acq_filters import has_n_coils, is_image_acquisition
from mrpro.data.AcqInfo import AcqInfo, convert_time_stamp_osi2, convert_time_stamp_siemens
from mrpro.data.EncodingLimits import EncodingLimits
from mrpro.data.enums import AcqFlags
from mrpro.data.KHeader import KHeader
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mrpro.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd
from mrpro.utils.typing import FileOrPath

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
    """K-space data. Shape `(*other coils k2 k1 k0)`"""

    traj: KTrajectory
    """K-space trajectory along kz, ky and kx. Shape `(*other k2 k1 k0)`"""

    @classmethod
    def from_file(
        cls,
        filename: FileOrPath,
        trajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: dict[str, object] | None = None,
        dataset_idx: int = -1,
        acquisition_filter_criterion: Callable = is_image_acquisition,
    ) -> Self:
        """Load k-space data from an ISMRMRD file.

        Parameters
        ----------
        filename
            path to the ISMRMRD file or file-like object
        trajectory
            KTrajectoryCalculator to calculate the k-space trajectory or an already calculated KTrajectory
            If a KTrajectory is given, the shape should be `(acquisisions 1 1 k0)` in the same order as the acquisitions
            in the ISMRMRD file.
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

        if ismrmrd_header.acquisitionSystemInformation is not None and isinstance(
            ismrmrd_header.acquisitionSystemInformation.systemVendor, str
        ):
            match ismrmrd_header.acquisitionSystemInformation.systemVendor.lower():
                case 'siemens':
                    convert_time_stamp = convert_time_stamp_siemens  # 2.5ms time steps
                case 'osi2':
                    convert_time_stamp = convert_time_stamp_osi2  # 1ms time steps
                case str(vendor):
                    warnings.warn(
                        f'Unknown vendor {vendor}. '
                        'Assuming Siemens time stamp format. If this is wrong, consider opening an Issue.',
                        stacklevel=1,
                    )
        else:
            warnings.warn('No vendor information found. Assuming Siemens time stamp format.', stacklevel=1)
            convert_time_stamp = convert_time_stamp_siemens

        acq_info, (k0_center, n_k0_tensor, discard_pre, discard_post) = AcqInfo.from_ismrmrd_acquisitions(
            acquisitions,
            additional_fields=('center_sample', 'number_of_samples', 'discard_pre', 'discard_post'),
            convert_time_stamp=convert_time_stamp,
        )

        if len(torch.unique(acq_info.idx.user5)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 5 indices for storing the kspace center line number.\n'
                'User 5 indices will be ignored',
                stacklevel=1,
            )

        if len(torch.unique(acq_info.idx.user6)) > 1:
            warnings.warn(
                'The Siemens to ismrmrd converter currently (ab)uses '
                'the user 6 indices for storing the kspace center partition number.\n'
                'User 6 indices will be ignored',
                stacklevel=1,
            )

        shapes = (torch.as_tensor([acq.data.shape[-1] for acq in acquisitions]) - discard_pre - discard_post).unique()
        if len(shapes) > 1:
            warnings.warn(
                f'Acquisitions have different shape. Got {list(shapes)}. '
                f'Keeping only acquisistions with {shapes[-1]} data samples. Note: discard_pre and discard_post '
                f'{"have been applied. " if discard_pre.any() or discard_post.any() else "were empty. "}'
                'Please open an issue of you need to handle this kind of data.',
                stacklevel=1,
            )
        data = torch.stack(
            [
                torch.as_tensor(acq.data[..., pre : acq.data.shape[-1] - post], dtype=torch.complex64)
                for acq, pre, post in zip(acquisitions, discard_pre, discard_post, strict=True)
                if acq.data.shape[-1] - pre - post == shapes[-1]
            ]
        )
        data = rearrange(data, 'acquisitions coils k0 -> acquisitions coils 1 1 k0')

        # Raises ValueError if required fields are missing in the header
        header = KHeader.from_ismrmrd(
            ismrmrd_header,
            acq_info,
            defaults={
                'datetime': modification_time,  # use the modification time of the dataset as fallback
                'trajectory': trajectory,
            },
            overwrite=header_overwrites,
        )
        # Calculate trajectory and check if it matches the kdata shape
        match trajectory:
            case KTrajectoryIsmrmrd():
                trajectory_ = trajectory(acquisitions, encoding_matrix=header.encoding_matrix)
            case KTrajectoryCalculator():
                reversed_readout_mask = (header.acq_info.flags[..., 0] & AcqFlags.ACQ_IS_REVERSE.value).bool()
                n_k0_unique = torch.unique(n_k0_tensor)
                if len(n_k0_unique) > 1:
                    raise ValueError(
                        'Trajectory can only be calculated for constant number of readout samples.\n'
                        f'Got unique values {list(n_k0_unique)}'
                    )
                encoding_limits = EncodingLimits.from_ismrmrd_header(ismrmrd_header)
                trajectory_ = trajectory(
                    n_k0=int(n_k0_unique[0]),
                    k0_center=k0_center,
                    k1_idx=header.acq_info.idx.k1,
                    k1_center=encoding_limits.k1.center,
                    k2_idx=header.acq_info.idx.k2,
                    k2_center=encoding_limits.k2.center,
                    reversed_readout_mask=reversed_readout_mask,
                    encoding_matrix=header.encoding_matrix,
                )
            case KTrajectory():
                try:
                    torch.broadcast_shapes(trajectory.broadcasted_shape, (data.shape[0], *data.shape[-3:]))
                except RuntimeError:
                    raise ValueError(
                        f'Trajectory shape {trajectory.broadcasted_shape} does not match data shape {data.shape}.'
                    ) from None
                trajectory_ = trajectory
            case _:
                raise TypeError(
                    'ktrajectory must be KTrajectoryIsmrmrd, KTrajectory or KTrajectoryCalculator'
                    f'not {type(trajectory)}',
                )

        kdata = cls(header, data, trajectory_)
        kdata = kdata.reshape_by_idx()
        return kdata

    def reshape_by_idx(self) -> Self:
        """Sort and reshape according to the acquisistion indices.

        Reshapes the data to ("all other labels", coils, k2, k1, k0).
        Within "all other labels", the order is determined by `KDIM_SORT_LABELS`.
        """
        # First, determine if we can split into k2 and k1 and how large these should be
        acq_indices_other = torch.stack([getattr(self.header.acq_info.idx, label) for label in OTHER_LABELS], dim=0)
        _, n_acqs_per_other = torch.unique(acq_indices_other, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of the label values in "other"
        n_acqs_per_other = torch.unique(n_acqs_per_other)

        acq_indices_other_k2 = torch.cat((acq_indices_other, self.header.acq_info.idx.k2.unsqueeze(0)), dim=0)
        _, n_acqs_per_other_and_k2 = torch.unique(acq_indices_other_k2, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of other **and k2**
        n_acqs_per_other_and_k2 = torch.unique(n_acqs_per_other_and_k2)

        if len(n_acqs_per_other_and_k2) == 1:
            # This is the most common case:
            # All other and k2 combinations have the same number of aquisistions which we can use as k1
            # to reshape to (other, coils, k2, k1, k0)
            n_k1 = n_acqs_per_other_and_k2[0]
            n_k2 = n_acqs_per_other[0] // n_k1
        elif len(n_acqs_per_other) == 1:
            # We cannot split the data along phase encoding steps into k2 and k1, as there are different numbers of k1.
            # But all "other labels" combinations have the same number of aquisisitions,
            # so we can reshape to (other, coils, k, 1, k0)
            n_k1 = 1
            n_k2 = n_acqs_per_other[0]  # total number of k encoding steps
        else:
            # For different other combinations we have different numbers of aquisistions,
            # so we can only reshape to (acquisitions, coils, 1, 1, k0)
            # This might be an user error.
            warnings.warn(
                f'There are different numbers of acquisistions in'
                'different combinations of labels {"/".join(OTHER_LABELS)}: \n'
                f'Found {n_acqs_per_other.tolist()}. \n'
                'The data will be reshaped to (acquisitions, coils, 1, 1, k0). \n'
                'This needs to be adjusted be reshaping for a successful reconstruction. \n'
                'If unintenional, this might be caused by wrong labels in the ismrmrd file or a wrong flag filter. \n'
                'To fix, it might be necessary to subset the data such that it can be reshaped to '
                '(other, coils, k2, k1, k0), or indexes needs to be fixed. \n'
                'After fixing the issue, call reshape_by_idx and consider recalculating the trajectory.',
                stacklevel=1,
            )
            n_k1 = 1
            n_k2 = 1

        # Second, determine the sorting order
        acq_indices = np.stack([getattr(self.header.acq_info.idx, label).ravel() for label in KDIM_SORT_LABELS], axis=0)
        sort_idx = np.lexsort(acq_indices)  # torch does not have lexsort as of pytorch 2.6 (March 2025)

        # Finally, reshape and sort the tensors in acqinfo and acqinfo.idx, and kdata.
        header = self.header.apply(
            lambda field: rearrange(
                cast(Rotation | torch.Tensor, rearrange(field, '... coils k2 k1 k0 -> (... k2 k1) coils k0'))[sort_idx],
                '(other k2 k1) coils k0 -> other coils k2 k1 k0',
                k1=n_k1,
                k2=n_k2,
                k0=1,
            )
            if isinstance(field, torch.Tensor | Rotation)
            else field
        )

        data = rearrange(
            rearrange(self.data, '... coils k2 k1 k0-> (... k2 k1) coils k0 ')[sort_idx],
            '(other k2 k1) coils k0 -> other coils k2 k1 k0',
            k1=n_k1,
            k2=n_k2,
        )

        kz, ky, kx = rearrange(
            self.traj.as_tensor(-1).flatten(end_dim=-4)[sort_idx],
            '(other k2 k1) coils k0 zyx -> zyx other coils k2 k1 k0 ',
            k1=n_k1,
            k2=n_k2,
        )
        traj = KTrajectory(kz, ky, kx, self.traj.grid_detection_tolerance, self.traj.repeat_detection_tolerance)
        return type(self)(header=header, data=data, traj=traj)

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
        `ValueError`
            If both batch_dims and joint_dims are defined.
        `ValuerError`
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

    def remove_readout_os(self: Self) -> Self:
        """Remove any oversampling along the readout direction.

        Removes oversampling along the readout direction by cropping the data
        to the size of the reconstruction matrix in image space [GAD]_.

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
        `ValueError`
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
        header.encoding_matrix.x = cropped_data.shape[-1]

        return type(self)(header, cropped_data, cropped_traj)

    def select_other_subset(
        self: Self,
        subset_idx: torch.Tensor,
        subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Select a subset from the other dimension of KData.

        Note: This function will be deprecated in the future.

        Parameters
        ----------
        kdata
            K-space data `(other coils k2 k1 k0)`
        subset_idx
            Index which elements of the other subset to use, e.g. phase 0,1,2 and 5
        subset_label
            Name of the other label, e.g. phase

        Returns
        -------
            K-space data `(other_subset coils k2 k1 k0)`

        Raises
        ------
        `ValueError`
            If the subset indices are not available in the data
        """
        # Flatten multi-dimensional other
        n_other = self.data.shape[:-4]  # Assume that data is not broadcasted along other
        header = self.header.apply(
            lambda field: rearrange(
                field.expand(*n_other, *field.shape[-4:]), '... coils k2 k1 k0->(...) coils k2 k1 k0'
            )
            if isinstance(field, torch.Tensor | Rotation)
            else field
        )
        traj = self.traj.as_tensor()
        traj = torch.broadcast_to(traj, (traj.shape[0], *self.data.shape[:-4], *traj.shape[-4:]))  # broadcast "other"
        traj = traj.flatten(start_dim=1, end_dim=-5)  # flatten "other" dimensions
        data = self.data.flatten(end_dim=-5)

        # Find elements in the subset
        label_idx = getattr(header.acq_info.idx, subset_label)
        if not all(el in torch.unique(label_idx) for el in subset_idx):
            raise ValueError('Subset indices are outside of the available index range')
        other_idx = torch.cat([torch.where(idx == label_idx[:, 0, 0, 0, 0])[0] for idx in subset_idx], dim=0)

        # Select subset
        header.acq_info.apply_(lambda field: field[other_idx] if isinstance(field, torch.Tensor | Rotation) else field)
        data = data[other_idx]
        traj = traj[:, other_idx]
        return type(self)(header, data, type(self.traj).from_tensor(traj))

    def split_k1_into_other(
        self,
        split_idx: torch.Tensor,
        other_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Based on an index tensor, split the data in e.g. phases.

        Parameters
        ----------
        split_idx
            2D index describing  the k1 points in each block to be moved to the other dimension
            `(other_split, k1_per_split)`
        other_label
            Label of other dimension, e.g. repetition, phase

        Returns
        -------
            K-space data with new shape `((other other_split) coils k2 k1_per_split k0)`

        """
        n_other_split, n_k1_per_split = split_idx.shape
        n_k1 = self.data.shape[-2]  # This assumes that data is not broadcasted along k1

        def split(data: RotationOrTensor) -> RotationOrTensor:
            # broadcast "k1"
            expanded = data.expand((*data.shape[:-2], n_k1, data.shape[-1]))
            # cast due to https://github.com/python/mypy/issues/10817
            return cast(RotationOrTensor, expanded[..., split_idx, :])

        data = rearrange(
            split(self.data.flatten(end_dim=-5)), 'other coils k2 other_split k1 k0->(other other_split) coils k2 k1 k0'
        )

        traj = self.traj.as_tensor()
        traj = torch.broadcast_to(traj, (traj.shape[0], *self.data.shape[:-4], *traj.shape[-4:]))  # broadcast "other"
        traj = traj.flatten(start_dim=1, end_dim=-5)  # flatten "other" dimensions
        traj = rearrange(split(traj), 'dim other coils k2 other_split k1 k0->dim (other other_split) coils k2 k1 k0')

        header = self.header.apply(
            lambda field: rearrange(
                split(  # type: ignore[type-var] # mypy does not recognize return type of rearrange here
                    rearrange(field, '... coils k2 k1 k0 -> (...) coils k2 k1 k0 ')  # flatten "other" dimensions
                ),
                'other coils k2 other_split k1 k0->(other other_split) coils k2 k1 k0',
            )
            if isinstance(field, Rotation | torch.Tensor)
            else field
        )

        new_idx = repeat(
            torch.arange(n_other_split),
            'other_split-> (other_split other) 1 k2 k1 1',
            other=data.shape[0] // n_other_split,
            k2=data.shape[-3],
            k1=n_k1_per_split,
        )
        setattr(header.acq_info.idx, other_label, new_idx)

        return type(self)(header, data, type(self.traj).from_tensor(traj))
