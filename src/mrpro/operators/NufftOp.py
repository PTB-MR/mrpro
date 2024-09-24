"""Fourier Operator."""

from abc import ABC
from collections.abc import Sequence
import functools
from typing import Callable, Self, TypeVar

import numpy as np
import torch
from torchkbnufft import KbNufft, KbNufftAdjoint

from mrpro.data._kdata.KData import KData
from mrpro.data.enums import TrajType
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator

TupleOfInts = TypeVar("TupleOfInts", tuple[int], tuple[int, int], tuple[int, int, int])

class NufftOp(LinearOperator, ABC):
    """Non uniform FFT Operator Base Class.

    Note: Consider using the general FourierOp instead of this class directly."""


    @property
    def grid_size(self)
        return [int(size * self.oversampling) for size in self.image_size]

    def __init__(
        self,
        omega: KTrajectory,
        image_size: TupleOfInts,
        dim: TupleOfInts,
        oversampling: float = 2.0,
        max_batch_size: int = 0,

    ) -> None:
        """Fourier Operator class.

        Parameters
        ----------
        omega
            prepared k-space trajectory with shape
            (batch_trajectory batch_data 3 kspace_points) for 3D
            (batch_trajectory batch_data 2 kspace_points) for 2D
            (batch_trajectory batch_data 1 kspace_points) for 1D
        image_size
            dimension of the reconstructed image
        oversampling
            oversampling used for interpolation in non-uniform FFTs
        max_batch_size
            maximum batch size for the NUFFT. If 0, an heuristic might be used
            or the batch size might not be limited.
        """
        super().__init__()
        if len(image_size)!=omega.shape[2]:
            raise ValueError("The number of image dimensions and the size of omega's third dimension must match.")
        if len(dim)!=omega.shape[2]:
            raise ValueError("The number of dimensions and the size of omega's third dimension must match.")
        self.image_size = image_size
        self.oversampling = oversampling
        self.register_buffer('_omega', omega)
        self.max_batch_size = max_batch_size


    def setup(self, data: torch.Tensor|None):
        """Setup the operator by creating plans etc.

        Should be called in both forward and adjoint methods,
        in case a change in the setup is required.

        Should do nothing if the setup is already done and valid.

        Parameters
        ----------
        data
            data that the operator will be applied to.
            If None, default values must be used for the setup.
            The Operator might not use the data,
            but it is passed to allow for data-dependent setup.
        """


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator mapping images to non-uniform k-space data.

        Parameters
        ----------
        x
            image data with shape
            (batch_trajectory batch_data z y x) if 3D,
            (batch_trajectory batch_data y x) if 2D,
            (batch_trajectory batch_data x) if 1D.


        Returns
        -------
             k-space data with shape: (batch_trajectory batch_data k2 k1 k0)
        """
        if not len(self.dim):
            return x,
        keep_dims = [i%x.ndim for i in self.dim]
        permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
        unpermute = np.argsort(permute)

        x = x.permute(*permute)
        permuted_x_shape = x.shape
        x = x.flatten(end_dim=-len(keep_dims) - 1)

            # omega should be (... non_nufft_dims) n_nufft_dims (nufft_dims)
            # TODO: consider moving the broadcast along fft dimensions to __init__ (independent of x shape).
            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._fwd_nufft_op(x, omega, norm='ortho')

            shape_nufft_dims = [self._kshape[i] for i in self._nufft_dims]
            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], -1, *shape_nufft_dims)  # -1 is coils
            x = x.permute(*unpermute)


    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator mapping the coil k-space data to the coil images.

        Parameters
        ----------
        x
            coil k-space data with shape:
            (batch_trajectory batch_data k2 k1 k0)

        Returns
        -------
            coil image data with shape: (... coils z y x)
        """
        if not len(self.dim):
            return x,


        x = self._adj_nufft_op(x, omega, norm='ortho')

            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], *x.shape[-len(keep_dims) :])
            x = x.permute(*unpermute)

        return (x,)

