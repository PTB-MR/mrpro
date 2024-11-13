"""Fourier Operator."""

from collections.abc import Sequence
from itertools import product

import numpy as np
import torch
from torchkbnufft import KbNufft, KbNufftAdjoint
from typing_extensions import Self

from mrpro.data._kdata.KData import KData
from mrpro.data.enums import TrajType
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.CartesianSamplingOp import CartesianSamplingOp
from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator


class FourierOp(LinearOperator, adjoint_as_backward=True):
    """Fourier Operator class."""

    def __init__(
        self,
        recon_matrix: SpatialDimension[int],
        encoding_matrix: SpatialDimension[int],
        traj: KTrajectory,
        nufft_oversampling: float = 2.0,
        nufft_numpoints: int = 6,
        nufft_kbwidth: float = 2.34,
    ) -> None:
        """Fourier Operator class.

        Parameters
        ----------
        recon_matrix
            dimension of the reconstructed image
        encoding_matrix
            dimension of the encoded k-space
        traj
            the k-space trajectories where the frequencies are sampled
        nufft_oversampling
            oversampling used for interpolation in non-uniform FFTs
        nufft_numpoints
            number of neighbors for interpolation in non-uniform FFTs
        nufft_kbwidth
            size of the Kaiser-Bessel kernel interpolation in non-uniform FFTs
        """
        super().__init__()

        self._fft_k210: list[int] = []
        """Dimensions in k-space data in which fft will be performed"""

        self._fft_kzyx: list[int] = []
        """Directions in which fft will be performed"""

        self._nufft_k210: list[int] = []
        """Dimensions in k-space data in which nufft will be performed"""

        self._nufft_kzyx: list[int] = []
        """Directions in which nufft will be performed"""

        self._singelton_k210: list[int] = []
        """Dimensions in k-space data that do not need to be transformed"""

        self._singelton__kzyx: list[int] = []
        """Directions that do not need to be transformed"""

        trajectory_types = traj.type_matrix
        type_along_k210 = np.bitwise_and.reduce(trajectory_types, 0)
        type_along_kzyx = np.bitwise_and.reduce(trajectory_types, 1)

        for dim_zyx, trajectory_type in zip((-3, -2, -1), type_along_kzyx, strict=True):
            if trajectory_type & TrajType.SINGLEVALUE.value:
                self._singelton__kzyx.append(dim_zyx)
            elif trajectory_type & TrajType.ONGRID.value:
                self._fft_kzyx.append(dim_zyx)
            else:
                self._nufft_kzyx.append(dim_zyx)
        for dim_210, trajectory_type in zip((-3, -2, -1), type_along_k210, strict=True):
            if trajectory_type & TrajType.SINGLEVALUE.value:
                self._singelton_k210.append(dim_210)
            elif trajectory_type & TrajType.ONGRID.value:
                self._fft_k210.append(dim_210)
            else:
                self._nufft_k210.append(dim_210)

        if self._fft_kzyx:  # need fft
            self._cart_sampling_op: CartesianSamplingOp | None = CartesianSamplingOp(
                encoding_matrix=encoding_matrix, traj=traj
            )
            self._fast_fourier_op: FastFourierOp | None = FastFourierOp(
                dim=self._fft_k210,
                recon_matrix=[recon_matrix.zyx[d] for d in self._fft_k210],
                encoding_matrix=[encoding_matrix.zyx[d] for d in self._fft_kzyx],
            )
        else:
            self._cart_sampling_op = None
            self._fast_fourier_op = None

        if self._nufft_kzyx:  # need nufft
            self._nufft_im_size = [recon_matrix.zyx[d] for d in self._nufft_kzyx]
            grid_size = [int(size * nufft_oversampling) for size in self._nufft_im_size]
            ks = [getattr(traj, ['kz', 'ky', 'kx'][d]) for d in self._nufft_kzyx]
            encoding_sizes = [encoding_matrix.zyx[d] for d in self._nufft_kzyx]
            omega = torch.broadcast_tensors(
                *(k * 2 * torch.pi / encoding_size for k, encoding_size in zip(ks, encoding_sizes, strict=True))
            )
            self.register_buffer('_omega', torch.stack(omega, dim=-4))  # use the 'coil' dim for the direction
            numpoints = [min(img_size, nufft_numpoints) for img_size in self._nufft_im_size]
            self._fwd_nufft_op: KbNufft | None = KbNufft(
                im_size=self._nufft_im_size,
                grid_size=grid_size,
                numpoints=numpoints,
                kbwidth=nufft_kbwidth,
            )
            self._adj_nufft_op: KbNufftAdjoint | None = KbNufftAdjoint(
                im_size=self._nufft_im_size,
                grid_size=grid_size,
                numpoints=numpoints,
                kbwidth=nufft_kbwidth,
            )
        else:
            self._omega: torch.Tensor | None = None
            self._fwd_nufft_op = None
            self._adj_nufft_op = None
        self._kshape = traj.broadcasted_shape

    @classmethod
    def from_kdata(cls, kdata: KData, recon_shape: SpatialDimension[int] | None = None) -> Self:
        """Create an instance of FourierOp from kdata with default settings.

        Parameters
        ----------
        kdata
            k-space data
        recon_shape
            dimension of the reconstructed image. Defaults to KData.header.recon_matrix
        """
        return cls(
            recon_matrix=kdata.header.recon_matrix if recon_shape is None else recon_shape,
            encoding_matrix=kdata.header.encoding_matrix,
            traj=kdata.traj,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Forward operator mapping the coil-images to the coil k-space data.

        Parameters
        ----------
        x
            coil image data with shape: (... coils z y x)

        Returns
        -------
            coil k-space data with shape: (... coils k2 k1 k0)
        """
        if self._fwd_nufft_op is not None and self._omega is not None:
            # NUFFT Type 2
            # we need to move the nufft-dimensions to the end and flatten all other dimensions
            # so the new shape will be (... non_nufft_dims) coils nufft_dims
            # we could move the permute to __init__ but then we still would need to prepend if len(other)>1
            keep_dims_zyx = [-4, *self._nufft_kzyx]  # -4 is always coil
            keep_dims_210 = [-4, *self._nufft_k210]  # -4 is always coil

            permute_zyx = [i for i in range(-x.ndim, 0) if i not in keep_dims_zyx] + keep_dims_zyx
            permute_210 = [i for i in range(-x.ndim, 0) if i not in keep_dims_210] + keep_dims_210

            unpermute = np.argsort(permute_210)

            x = x.permute(*permute_zyx)
            unflatten_shape = x.shape[: -len(keep_dims_zyx)]
            x = x.flatten(end_dim=-len(keep_dims_zyx) - 1)

            # omega should be (... non_nufft_dims) n_nufft_dims (nufft_dims)
            omega = self._omega.permute(*permute_210)
            omega = omega.broadcast_to(*unflatten_shape, *omega.shape[-len(keep_dims_zyx) :])
            omega = omega.flatten(end_dim=-len(keep_dims_210) - 1).flatten(start_dim=-len(keep_dims_210) + 1)

            x = self._fwd_nufft_op(x, omega, norm='ortho')
            shape_nufft_dims = [self._kshape[i] for i in self._nufft_k210]
            nufft_singletons = [1] * (len(self._nufft_kzyx) - len(self._nufft_k210))
            x = x.reshape(*unflatten_shape, *nufft_singletons, -1, *shape_nufft_dims)  # -1 is coils
            x = x.permute(*unpermute)

        if self._fast_fourier_op is not None and self._cart_sampling_op is not None:
            # FFT
            (x,) = self._cart_sampling_op(self._fast_fourier_op(x)[0])

        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint operator mapping the coil k-space data to the coil images.

        Parameters
        ----------
        x
            coil k-space data with shape: (... coils k2 k1 k0)

        Returns
        -------
            coil image data with shape: (... coils z y x)
        """
        if self._fast_fourier_op is not None and self._cart_sampling_op is not None:
            # IFFT
            (x,) = self._fast_fourier_op.adjoint(self._cart_sampling_op.adjoint(x)[0])

        if self._adj_nufft_op is not None and self._omega is not None:
            # NUFFT Type 1
            # we need to move the nufft-dimensions to the end, flatten them and flatten all other dimensions
            # so the new shape will be (... non_nufft_dims) coils (nufft_dims)
            keep_dims_210 = [-4, *self._nufft_k210]  # -4 is coil
            permute_210 = [i for i in range(-x.ndim, 0) if i not in keep_dims_210] + keep_dims_210

            keep_dims_zyx = [-4, *self._nufft_kzyx]
            unpermute = np.argsort([i for i in range(-x.ndim, 0) if i not in keep_dims_zyx] + keep_dims_zyx)

            x = x.permute(*permute_210)
            unflatten_shape = x.shape[: -len(keep_dims_zyx)]
            x = x.flatten(end_dim=-len(keep_dims_210) - 1).flatten(start_dim=-len(keep_dims_210) + 1)

            omega = self._omega.permute(*permute_210)
            omega = omega.broadcast_to(*unflatten_shape, *omega.shape[-len(keep_dims_zyx) :])
            omega = omega.flatten(end_dim=-len(keep_dims_210) - 1).flatten(start_dim=-len(keep_dims_210) + 1)

            x = self._adj_nufft_op(x, omega, norm='ortho')

            x = x.reshape(*unflatten_shape, -1, *x.shape[-len(self._nufft_kzyx) :])  # -1 is coils
            x = x.permute(*unpermute)

        return (x,)

    @property
    def gram(self) -> LinearOperator:
        """Return the gram operator."""
        return FourierGramOp(self)


def symmetrize(kernel: torch.Tensor, rank: int) -> torch.Tensor:
    """Enforce hermitian symmetry on the kernel. Returns only half of the kernel."""
    flipped = kernel.clone()
    for d in range(-rank, 0):
        flipped = flipped.index_select(d, -1 * torch.arange(flipped.shape[d], device=flipped.device) % flipped.size(d))
    kernel = (kernel + flipped.conj()) / 2
    last_len = kernel.shape[-1]
    return kernel[..., : last_len // 2 + 1]


def gram_nufft_kernel(weight: torch.Tensor, trajectory: torch.Tensor, recon_shape: Sequence[int]) -> torch.Tensor:
    """Calculate the convolution kernel for the NUFFT gram operator.

    Parameters
    ----------
    weight
        either ones or density compensation weights
    trajectory
        k-space trajectory
    recon_shape
        shape of the reconstructed image

    Returns
    -------
    kernel
        real valued convolution kernel for the NUFFT gram operator, already in Fourier space
    """
    rank = trajectory.shape[-2]
    if rank != len(recon_shape):
        raise ValueError('Rank of trajectory and image size must match.')
    # Instead of doing one adjoint nufft with double the recon size in all dimensions,
    # we do two adjoint nuffts per dimensions, saving a lot of memory.
    adjnufft_ob = KbNufftAdjoint(im_size=recon_shape, n_shift=[0] * rank).to(trajectory)

    kernel = adjnufft_ob(weight, trajectory)  # this will be the top left ... corner block
    pad = []
    for s in kernel.shape[: -rank - 1 : -1]:
        pad.extend([0, s])
    kernel = torch.nn.functional.pad(kernel, pad)  # twice the size in all dimensions

    for flips in list(product([1, -1], repeat=rank)):
        if all(flip == 1 for flip in flips):
            # top left ... block already processed before padding
            continue
        flipped_trajectory = trajectory * torch.tensor(flips).to(trajectory).unsqueeze(-1)
        kernel_part = adjnufft_ob(weight, flipped_trajectory)
        slices = []  # which part of the kernel to is currently being processed
        for dim, flip in zip(range(-rank, 0), flips, strict=True):
            if flip > 0:  # first half in the dimension
                slices.append(slice(0, kernel_part.size(dim)))
            else:  # second half in the dimension
                slices.append(slice(kernel_part.size(dim) + 1, None))
                kernel_part = kernel_part.index_select(dim, torch.arange(kernel_part.size(dim) - 1, 0, -1))  # flip

        kernel[[..., *slices]] = kernel_part

    kernel = symmetrize(kernel, rank)
    kernel = torch.fft.hfftn(kernel, dim=list(range(-rank, 0)), norm='backward')
    kernel /= kernel.shape[-rank:].numel()
    kernel = torch.fft.fftshift(kernel, dim=list(range(-rank, 0)))
    return kernel


class FourierGramOp(LinearOperator):
    """Gram operator for the Fourier operator.

    Implements the adjoint of the forward operator of the Fourier operator, i.e. the gram operator
    `F.H@F.

    Uses a convolution, implemented as multiplication in Fourier space, to calculate the gram operator
    for the toeplitz NUFFT operator.

    Uses a multiplication with a binary mask in Fourier space to calculate the gram operator for
    the Cartesian FFT operator

    This Operator is only used internally and should not be used directly.
    Instead, consider using the `gram` property of :class: `mrpro.operators.FourierOp`.
    """

    _kernel: torch.Tensor | None

    def __init__(self, fourier_op: FourierOp) -> None:
        """Initialize the gram operator.

        If density compensation weights are provided, they the operator
        F.H@dcf@F is calculated.

        Parameters
        ----------
        fourier_op
            the Fourier operator to calculate the gram operator for

        """
        super().__init__()
        if fourier_op._nufft_k210 and fourier_op._omega is not None:
            # NUFFT Gram
            weight = torch.ones_like(fourier_op._omega[..., :1, :, :, :])
            keep_dims_zyx = [-4, *fourier_op._nufft_kzyx]  # -4 is coil
            permute = [i for i in range(-weight.ndim, 0) if i not in keep_dims_zyx] + keep_dims_zyx
            unpermute = np.argsort(permute)
            weight = weight.permute(*permute)
            weight_unflattend_shape = weight.shape
            weight = weight.flatten(end_dim=-len(keep_dims_zyx) - 1).flatten(start_dim=-len(keep_dims_zyx) + 1)
            weight = weight + 0j
            omega = fourier_op._omega.permute(*permute)
            omega = omega.flatten(end_dim=-len(keep_dims_zyx) - 1).flatten(start_dim=-len(keep_dims_zyx) + 1)
            kernel = gram_nufft_kernel(weight, omega, fourier_op._nufft_im_size)
            kernel = kernel.reshape(
                *weight_unflattend_shape[: -len(keep_dims_zyx)], *kernel.shape[-len(keep_dims_zyx) :]
            )
            kernel = kernel.permute(*unpermute)
            fft = FastFourierOp(
                dim=fourier_op._nufft_kzyx,
                encoding_matrix=[2 * s for s in fourier_op._nufft_im_size],
                recon_matrix=fourier_op._nufft_im_size,
            )
            self.nufft_gram: None | LinearOperator = fft.H * kernel @ fft
        else:
            self.nufft_gram = None

        if fourier_op._fast_fourier_op is not None and fourier_op._cart_sampling_op is not None:
            # FFT Gram
            self.fast_fourier_gram: None | LinearOperator = (
                fourier_op._fast_fourier_op.H @ fourier_op._cart_sampling_op.gram @ fourier_op._fast_fourier_op
            )
        else:
            self.fast_fourier_gram = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape (..., coils, z, y, x)
        """
        if self.nufft_gram is not None:
            (x,) = self.nufft_gram(x)

        if self.fast_fourier_gram is not None:
            (x,) = self.fast_fourier_gram(x)
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint operator to the input tensor.

        Parameters
        ----------
        x
            input tensor, shape (..., coils, k2, k1, k0)
        """
        return self.forward(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self
