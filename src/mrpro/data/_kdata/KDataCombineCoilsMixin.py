"""Combine receiver coils based on a PCA compression."""

from collections.abc import Sequence
from types import EllipsisType

import numpy as np
import torch
from typing_extensions import Self

from mrpro.data._kdata.KDataProtocol import _KDataProtocol


class KDataCombineCoilsMixin(_KDataProtocol):
    """Remove oversampling along readout dimension."""

    def combine_coils(
        self: Self,
        n_combined_coils: int,
        batch_dims: None | Sequence[int] = None,
        joint_dims: Sequence[int] | EllipsisType = ...,
    ) -> Self:
        """Reduce the number of coils based on a PCA compression.

        A PCA is carried out along the coil dimension and the n_combined_coils virtual coil elements are selected. For
        more information on coil combination and coil compression please see [BUE2007]_ and [DON2008]_.

        Returns a copy of the data.

        Parameters
        ----------
        kdata
            K-space data
        n_combined_coils
            Number of combined coils
        batch_dims
            Dimensions which are treated as batched, i.e. separate coil combination matrizes (e.g. different slices).
            Default is to do one coil combination matrix for the entire k-space data. Only batch_dim or joint_dim can
            be defined. If batch_dims is not None then joint_dims has to be ...
        joint_dims
            Dimensions which are combined to calculate single coil combination matrix (e.g. k0, k1, contrast). Default
            is that all dimensions (except for the coil dimension) are joint_dims. Only batch_dim or joint_dim can
            be defined. If joint_dims is not ... batch_dims has to be None

        Returns
        -------
            Copy of K-space data with combined coils.

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
        """
        from mrpro.operators import PCACompressionOp

        coil_dim = -4 % self.data.ndim
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
            batch_dims_normalized
            + [i for i in range(self.data.ndim) if i != coil_dim and i not in batch_dims_normalized]
            + [coil_dim]
        )
        kdata_coil_combined = self.data.permute(permute_order)
        permuted_kdata_shape = kdata_coil_combined.shape
        kdata_coil_combined = kdata_coil_combined.flatten(
            start_dim=len(batch_dims_normalized), end_dim=-2
        )  # keep separate dimensions and coil

        pca_compression_op = PCACompressionOp(data=kdata_coil_combined, n_components=n_combined_coils)
        (kdata_coil_combined,) = pca_compression_op(kdata_coil_combined)

        # reshape to original dimensions and undo permutation
        kdata_coil_combined = torch.reshape(
            kdata_coil_combined, [*permuted_kdata_shape[:-1], n_combined_coils]
        ).permute(*np.argsort(permute_order))

        return type(self)(self.header, kdata_coil_combined, self.traj)
