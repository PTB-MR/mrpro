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

        def get_spatial_dims(spatial_dims: SpatialDimension, dims: Sequence[int]):
            return [
                s
                for s, i in zip((spatial_dims.z, spatial_dims.y, spatial_dims.x), (-3, -2, -1), strict=True)
                if i in dims
            ]

        def get_traj(traj: KTrajectory, dims: Sequence[int]):
            return [k for k, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1), strict=True) if i in dims]

        self._ignore_dims, self._fft_dims, self._nufft_dims = [], [], []
        for dim, type_ in zip((-3, -2, -1), traj.type_along_kzyx, strict=True):
            if type_ & TrajType.SINGLEVALUE:
                # dimension which do not require any transform
                self._ignore_dims.append(dim)
            elif type_ & TrajType.ONGRID:
                self._fft_dims.append(dim)
            else:
                self._nufft_dims.append(dim)

        if self._fft_dims:
            self._fast_fourier_op: FastFourierOp | None = FastFourierOp(
                dim=tuple(self._fft_dims),
                recon_matrix=get_spatial_dims(recon_matrix, self._fft_dims),
                encoding_matrix=get_spatial_dims(encoding_matrix, self._fft_dims),
            )
            self._cart_sampling_op: CartesianSamplingOp | None = CartesianSamplingOp(
                encoding_matrix=encoding_matrix, traj=traj
            )
        else:
            self._fast_fourier_op = None
            self._cart_sampling_op = None
        # Find dimensions which require NUFFT
        if self._nufft_dims:
            fft_dims_k210 = [
                dim
                for dim in (-3, -2, -1)
                if (traj.type_along_k210[dim] & TrajType.ONGRID)
                and not (traj.type_along_k210[dim] & TrajType.SINGLEVALUE)
            ]
            if self._fft_dims != fft_dims_k210:
                raise NotImplementedError(
                    'If both FFT and NUFFT dims are present, Cartesian FFT dims need to be aligned with the '
                    'k-space dimension, i.e. kx along k0, ky along k1 and kz along k2',
                )

            self._nufft_im_size = get_spatial_dims(recon_matrix, self._nufft_dims)
            grid_size = [int(size * nufft_oversampling) for size in self._nufft_im_size]
            omega = [
                k * 2 * torch.pi / ks
                for k, ks in zip(
                    get_traj(traj, self._nufft_dims),
                    get_spatial_dims(encoding_matrix, self._nufft_dims),
                    strict=True,
                )
            ]

            # Broadcast shapes not always needed but also does not hurt
            omega = [k.expand(*np.broadcast_shapes(*[k.shape for k in omega])) for k in omega]
            self.register_buffer('_omega', torch.stack(omega, dim=-4))  # use the 'coil' dim for the direction
            numpoints = [min(img_size, nufft_numpoints) for img_size in self._nufft_im_size]
            self._fwd_nufft_op: KbNufftAdjoint | None = KbNufft(
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
            keep_dims = [-4, *self._nufft_dims]  # -4 is always coil
            permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)

            x = x.permute(*permute)
            permuted_x_shape = x.shape
            x = x.flatten(end_dim=-len(keep_dims) - 1)

            # omega should be (... non_nufft_dims) n_nufft_dims (nufft_dims)
            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._fwd_nufft_op(x, omega, norm='ortho')

            shape_nufft_dims = [self._kshape[i] for i in self._nufft_dims]
            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], -1, *shape_nufft_dims)  # -1 is coils
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
            keep_dims = [-4, *self._nufft_dims]  # -4 is coil
            permute = [i for i in range(-x.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)

            x = x.permute(*permute)
            permuted_x_shape = x.shape
            x = x.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            omega = self._omega.permute(*permute)
            omega = omega.broadcast_to(*permuted_x_shape[: -len(keep_dims)], *omega.shape[-len(keep_dims) :])
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)

            x = self._adj_nufft_op(x, omega, norm='ortho')

            x = x.reshape(*permuted_x_shape[: -len(keep_dims)], *x.shape[-len(keep_dims) :])
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
        if fourier_op._nufft_dims and fourier_op._omega is not None:
            weight = torch.ones_like(fourier_op._omega[..., :1, :, :, :])
            keep_dims = [-4, *fourier_op._nufft_dims]  # -4 is coil
            permute = [i for i in range(-weight.ndim, 0) if i not in keep_dims] + keep_dims
            unpermute = np.argsort(permute)
            weight = weight.permute(*permute)
            weight_unflattend_shape = weight.shape
            weight = weight.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)
            weight = weight + 0j
            omega = fourier_op._omega.permute(*permute)
            omega = omega.flatten(end_dim=-len(keep_dims) - 1).flatten(start_dim=-len(keep_dims) + 1)
            kernel = gram_nufft_kernel(weight, omega, fourier_op._nufft_im_size)
            kernel = kernel.reshape(*weight_unflattend_shape[: -len(keep_dims)], *kernel.shape[-len(keep_dims) :])
            kernel = kernel.permute(*unpermute)
            fft = FastFourierOp(
                dim=fourier_op._nufft_dims,
                encoding_matrix=[2 * s for s in fourier_op._nufft_im_size],
                recon_matrix=fourier_op._nufft_im_size,
            )
            self.nufft_gram: None | LinearOperator = fft.H * kernel @ fft
        else:
            self.nufft_gram = None

        if fourier_op._fast_fourier_op is not None and fourier_op._cart_sampling_op is not None:
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
