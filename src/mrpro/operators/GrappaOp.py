from typing import overload

import torch
from einops import rearrange

from mrpro.data.enums import TrajType
from mrpro.data.KData import KData
from mrpro.operators.CartesianSamplingOp import CartesianSamplingOp
from mrpro.operators.Operator import Operator
from mrpro.utils.sliding_window import sliding_window


class GrappaOp(Operator[torch.Tensor, torch.Tensor]):
    """GRAPPA operator for filling in missing k-space points."""

    def __init__(
        self,
        acs_kdata: KData,
        sampling_op: CartesianSamplingOp,
        kernel_size: tuple[int, ...] = (5, 5),
        grappa_dims: tuple[int, ...] = (-2, -1),
        regularizer: float = 1e-5,
    ) -> None:
        """Initialize a GRAPPA operator.

        Parameters
        ----------
        acs_kdata
            The ACS data to use for GRAPPA. Can be obtained by indexing the aquired KData.
        sampling_op
            The sampling operator for the undersampled data. Can be obtaineded by CartesianSamplingOp.from_kdata.
        kernel_size
            The size of the kernel to use for GRAPPA.
        grappa_dims
            The dimensions to use for GRAPPA.
        regularizer
            The regularization parameter for GRAPPA.
        """
        super().__init__()

        if not all(acs_kdata.traj.type_along_kzyx[dim] == TrajType.ONGRID for dim in grappa_dims):
            raise ValueError(f'ACS data for GRAPPA must be Cartesian in dimensions {grappa_dims}.')
        if len(kernel_size) != len(grappa_dims):
            raise ValueError('Length of kernel_size must match length of grappa_dims.')

        self.encoding_matrix = acs_kdata.header.encoding_matrix
        self.grappa_dims = grappa_dims
        acs_data = acs_kdata.data
        num_coils = acs_data.shape[-4]

        source_patches = sliding_window(acs_data, window_shape=kernel_size, dim=self.grappa_dims)

        num_features = num_coils * torch.prod(torch.tensor(kernel_size)).item()
        source_matrix = source_patches.reshape(-1, num_features)

        center_indices = tuple(k // 2 for k in kernel_size)
        index = (..., slice(None), *tuple([slice(None)] * (source_patches.ndim - acs_data.ndim)), *center_indices)
        target_matrix = source_patches[index].reshape(-1, num_coils)

        shs = source_matrix.mH @ source_matrix
        reg_term = regularizer * torch.eye(shs.shape[0], device=shs.device, dtype=shs.dtype)
        sht = source_matrix.mH @ target_matrix
        weights = torch.linalg.solve(shs + reg_term, sht)

        self.grappa_kernel = rearrange(weights, '(c_in ...) c_out -> c_out c_in ...', c_in=num_coils).to(acs_data.dtype)
        self.sampling_op = sampling_op

    def _prepare_for_conv(self, data: torch.Tensor):
        ndim = data.ndim
        coil_dim = ndim - 4
        grappa_dims_pos = [d % ndim for d in self.grappa_dims]
        other_dims = sorted([d for d in range(ndim) if d not in grappa_dims_pos and d != coil_dim])

        permute_order = (*other_dims, coil_dim, *grappa_dims_pos)
        permuted_data = data.permute(*permute_order)

        num_other_dims = len(other_dims)
        if num_other_dims > 0:
            data_to_conv = permuted_data.flatten(start_dim=0, end_dim=num_other_dims - 1)
        else:
            data_to_conv = permuted_data

        original_other_shape = tuple(data.shape[d] for d in other_dims)
        return data_to_conv, permute_order, original_other_shape

    def _unprepare_from_conv(
        self,
        data_conv: torch.Tensor,
        permute_order: list[int],
        original_other_shape: tuple[int, ...],
    ):
        if len(original_other_shape) > 0:
            unflattened_data = data_conv.unflatten(0, original_other_shape)
        else:
            unflattened_data = data_conv

        inverse_permute_order = torch.argsort(torch.tensor(permute_order))
        return unflattened_data.permute(*inverse_permute_order)

    @overload
    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor,]: ...

    @overload
    def forward(self, data: KData) -> KData: ...

    def forward(self, data: torch.Tensor | KData) -> tuple[torch.Tensor,] | KData:
        """Apply GRAPPA to the data, i.e. fill in the missing data points.

        Parameters
        ----------
        data
            The data to apply GRAPPA to.
        """
        if isinstance(data, KData):
            (gridded_data,) = self.sampling_op.adjoint(data.data)
        else:
            (gridded_data,) = self.sampling_op.adjoint(data)

        (
            data_to_conv,
            permute_order,
            original_other_shape,
        ) = self._prepare_for_conv(gridded_data)

        conv_fn = (torch.nn.functional.conv1d, torch.nn.functional.conv2d, torch.nn.functional.conv3d)[
            len(self.grappa_dims) - 1
        ]
        synthesized_flat = conv_fn(data_to_conv, self.grappa_kernel, padding='same')
        synthesized_data = self._unprepare_from_conv(synthesized_flat, permute_order, original_other_shape)

        (masked_synthesized,) = self.sampling_op.gram(synthesized_data)
        reconstructed_data = gridded_data + synthesized_data - masked_synthesized

        if isinstance(data, KData):
            header = data.header.clone()
            full_traj = self.sampling_op.full_trajectory()
            return KData(header=header, data=reconstructed_data, traj=full_traj)
        return (reconstructed_data,)
