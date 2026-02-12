"""MR raw data / k-space data class."""

import copy
import dataclasses
import datetime
import warnings
from collections.abc import Callable, Mapping, Sequence
from importlib.metadata import version
from types import EllipsisType
from typing import Literal, cast

import h5py
import ismrmrd
import numpy as np
import torch
from einops import rearrange
from typing_extensions import Self, TypeVar

from mr2.data.acq_filters import has_n_coils, is_image_acquisition
from mr2.data.AcqInfo import (
    AcqInfo,
    convert_time_stamp_from_osi2,
    convert_time_stamp_from_siemens,
    convert_time_stamp_to_osi2,
    convert_time_stamp_to_siemens,
    write_acqinfo_to_ismrmrd_acquisition_,
)
from mr2.data.Dataclass import Dataclass
from mr2.data.EncodingLimits import EncodingLimits, Limits
from mr2.data.enums import AcqFlags
from mr2.data.KHeader import KHeader
from mr2.data.KTrajectory import KTrajectory
from mr2.data.Rotation import Rotation
from mr2.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator
from mr2.data.traj_calculators.KTrajectoryIsmrmrd import KTrajectoryIsmrmrd
from mr2.utils.summarize import summarize_object
from mr2.utils.typing import FileOrPath

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

T = TypeVar('T', Rotation, torch.Tensor, Rotation | torch.Tensor)


class KData(Dataclass):
    """MR raw data / k-space data class."""

    header: KHeader
    """Header information for k-space data"""

    data: torch.Tensor
    """K-space data. Shape `(*other coils k2 k1 k0)`"""

    traj: KTrajectory
    """K-space trajectory along kz, ky and kx. Shape `(*other 1 k2 k1 k0)`"""

    @classmethod
    def from_file(
        cls,
        filename: FileOrPath,
        trajectory: KTrajectoryCalculator | KTrajectory | KTrajectoryIsmrmrd,
        header_overwrites: Mapping[str, object] | None = None,
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
                    convert_time_stamp = convert_time_stamp_from_siemens  # 2.5ms time steps
                case 'osi2':
                    convert_time_stamp = convert_time_stamp_from_osi2  # 1ms time steps
                case str(vendor):
                    warnings.warn(
                        f'Unknown vendor {vendor}. '
                        'Assuming Siemens time stamp format. If this is wrong, consider opening an Issue.',
                        stacklevel=1,
                    )
                    convert_time_stamp = convert_time_stamp_from_siemens  # 2.5ms time steps
        else:
            warnings.warn('No vendor information found. Assuming Siemens time stamp format.', stacklevel=1)
            convert_time_stamp = convert_time_stamp_from_siemens

        acq_info, (k0_center, n_k0_tensor, discard_pre, discard_post) = AcqInfo.from_ismrmrd_acquisitions(
            acquisitions,
            additional_fields=('center_sample', 'number_of_samples', 'discard_pre', 'discard_post'),
            convert_time_stamp=convert_time_stamp,
        )
        discard_pre_1d = rearrange(discard_pre, 'n_readouts 1 1 1 1 -> n_readouts')
        discard_post_1d = rearrange(discard_post, 'n_readouts 1 1 1 1 -> n_readouts')

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

        shapes = (
            torch.as_tensor([acq.data.shape[-1] for acq in acquisitions]) - discard_pre_1d - discard_post_1d
        ).unique()
        if len(shapes) > 1:
            warnings.warn(
                f'Acquisitions have different shape. Got {list(shapes)}. '
                f'Keeping only acquisistions with {shapes[-1]} data samples. Note: discard_pre and discard_post '
                f'{"have been applied. " if discard_pre_1d.any() or discard_post_1d.any() else "were empty. "}'
                'Please open an issue of you need to handle this kind of data.',
                stacklevel=1,
            )
        data = torch.stack(
            [
                torch.as_tensor(acq.data[..., pre : acq.data.shape[-1] - post], dtype=torch.complex64)
                for acq, pre, post in zip(acquisitions, discard_pre_1d, discard_post_1d, strict=True)
                if acq.data.shape[-1] - pre - post == shapes[-1]
            ]
        )
        n_k0_tensor = n_k0_tensor - discard_pre - discard_post
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
        if header.vendor.lower() == 'siemens':
            # Siemens assumes a fft to go from k-space to image space
            data = data.conj_physical()

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
                    torch.broadcast_shapes(trajectory.shape, (data.shape[0], *data.shape[-3:]))
                except RuntimeError:
                    raise ValueError(
                        f'Trajectory shape {trajectory.shape} does not match data shape {data.shape}.'
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
        shape = self.shape

        def expand(x: T) -> T:
            x = x.expand(*shape[:-4], -1, shape[-3], shape[-2], -1)
            return x

        # First, determine if we can split into k2 and k1 and how large these should be
        acq_indices_other = torch.stack(
            [expand(getattr(self.header.acq_info.idx, label)).ravel() for label in OTHER_LABELS],
            dim=0,
        )
        _, n_acqs_per_other = torch.unique(acq_indices_other, dim=1, return_counts=True)
        # unique counts of acquisitions for each combination of the label values in "other"
        n_acqs_per_other = torch.unique(n_acqs_per_other)

        acq_indices_other_k2 = torch.cat(
            (acq_indices_other, expand(self.header.acq_info.idx.k2).ravel().unsqueeze(0)), dim=0
        )
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
        acq_indices = np.stack(
            [expand(getattr(self.header.acq_info.idx, label)).ravel() for label in KDIM_SORT_LABELS],
            axis=0,
        )
        sort_idx = torch.as_tensor(np.lexsort(acq_indices))  # torch has no lexsort as of pytorch 2.6 (March 2025)

        # Finally, reshape and sort the tensors in acqinfo and acqinfo.idx, and kdata.
        def sort(x: T) -> T:
            flat = cast(T, rearrange(expand(x), '... coils k2 k1 k0 -> (... k2 k1) coils k0'))
            return cast(
                T, rearrange(flat[sort_idx], '(other k2 k1) coils k0 -> other coils k2 k1 k0', k1=n_k1, k2=n_k2)
            )

        header = self.header.apply(lambda field: sort(field) if isinstance(field, torch.Tensor | Rotation) else field)
        data = sort(self.data)
        kz, ky, kx = (sort(t) for t in (self.traj.kz, self.traj.ky, self.traj.kx))
        traj = KTrajectory(kz, ky, kx, self.traj.grid_detection_tolerance, self.traj.repeat_detection_tolerance)
        return type(self)(header=header._reduce_repeats_(), data=data, traj=traj._reduce_repeats_())

    def to_file(self, filename: FileOrPath) -> None:
        """Save KData as ISMRMRD dataset to file.

        Parameters
        ----------
        filename
            path to the ISMRMRD file
        """
        ismrmrd_version = int(version('ismrmrd').split('.')[0])
        header = self.header.to_ismrmrd()
        header.acquisitionSystemInformation.receiverChannels = self.data.shape[-4]

        # Calculate the encoding limits as min/max of the acquisition indices
        def limits_from_acq_idx(acq_idx_tensor: torch.Tensor) -> Limits:
            return Limits(int(acq_idx_tensor.min().item()), int(acq_idx_tensor.max().item()), 0)

        encoding_limits = EncodingLimits(
            **{
                field.name: limits_from_acq_idx(getattr(self.header.acq_info.idx, field.name))
                for field in dataclasses.fields(self.header.acq_info.idx)
            }
        )

        # For the k-space center of k1 and k2 we can only make an educated guess on where it is:
        # k-space point closest to 0
        center_idx = self.traj.as_tensor(-1).abs().sum(dim=-1).argmin()
        encoding_limits.k1.center = int(self.header.acq_info.idx.k1.broadcast_to(self.traj.shape).flatten()[center_idx])
        encoding_limits.k2.center = int(self.header.acq_info.idx.k2.broadcast_to(self.traj.shape).flatten()[center_idx])
        header.encoding[0].encodingLimits = encoding_limits.to_ismrmrd_encoding_limits_type()

        # Vendors use different units for time stamps
        if self.header.vendor.lower() == 'osi2':
            convert_time_stamp = convert_time_stamp_to_osi2  # 1ms time steps
        else:
            convert_time_stamp = convert_time_stamp_to_siemens  # 2.5ms time steps

        with ismrmrd.Dataset(filename, 'dataset', create_if_needed=True) as dataset:
            dataset.write_xml_header(header.toXML('utf-8'))

            flattened = self.rearrange('... coils k2 k1 k0 -> (... k2 k1) coils 1 1 k0')
            for scan_counter, acq in enumerate(flattened):
                ismrmrd_acq = ismrmrd.Acquisition()
                ismrmrd_acq.resize(
                    number_of_samples=acq.shape[-1], active_channels=acq.shape[-4], trajectory_dimensions=3
                )
                ismrmrd_acq.available_channels = acq.shape[-4]
                ismrmrd_acq.version = ismrmrd_version
                ismrmrd_acq.scan_counter = 0
                write_acqinfo_to_ismrmrd_acquisition_(acq.header.acq_info, ismrmrd_acq, convert_time_stamp)
                ismrmrd_acq.traj[:] = acq.traj.as_tensor(-1).squeeze().cpu().numpy()[:, ::-1]  # zyx -> xyz
                ismrmrd_acq.center_sample = np.argmin(np.abs(ismrmrd_acq.traj[:, 0]))
                acq_data = acq.data.squeeze().cpu().numpy()
                # Siemens assumes a fft to go from k-space to image space
                ismrmrd_acq.data[:] = acq_data.conj() if self.header.vendor.lower() == 'siemens' else acq_data
                ismrmrd_acq.scan_counter = scan_counter
                dataset.append_acquisition(ismrmrd_acq)

    def __repr__(self):
        """Representation method for KData class."""
        traj_info = '\n   '.join(repr(self.traj).splitlines())
        header_info = '\n   '.join(repr(self.header).splitlines())
        representation = '\n'.join(
            [
                super().__repr__().splitlines()[0],
                f'  data: {summarize_object(self.data)}',
                f'  traj: {traj_info}',
                f'  header:  {header_info}',
            ]
        )
        return representation

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
        from mr2.operators import PCACompressionOp

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
        from mr2.operators.FastFourierOp import FastFourierOp

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
        header.encoding_fov.x = self.header.recon_fov.x

        return type(self)(header, cropped_data, cropped_traj)

    def select_other_subset(
        self: Self,
        subset_idx: torch.Tensor,
        subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Select a subset from the other dimension of KData.

        .. warning::
            This function will be deprecated in the future.

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
