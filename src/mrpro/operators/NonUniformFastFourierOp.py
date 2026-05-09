"""Non-Uniform Fast Fourier Operator."""

from collections.abc import Sequence
from functools import partial
from itertools import product
from types import EllipsisType
from typing import Literal

import einops
import numpy as np
import torch
from pytorch_finufft.functional import finufft_type1, finufft_type2
from typing_extensions import Self

from mrpro.data.DcfData import DcfData
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.PCACompressionOp import PCACompressionOp
from mrpro.utils import normalize_index, unsqueeze_right


class NonUniformFastFourierOp(LinearOperator, adjoint_as_backward=True):
    """Non-Uniform Fast Fourier Operator class."""

    def __init__(
        self,
        direction: Sequence[Literal['x', 'y', 'z', -3, -2, -1]],
        recon_matrix: SpatialDimension[int] | Sequence[int],
        encoding_matrix: SpatialDimension[int] | Sequence[int],
        traj: KTrajectory,
        oversampling: float = 2.0,
    ) -> None:
        """Initialize Non-Uniform Fast Fourier Operator.

        .. note::
            Consider using `~mrpro.operators.FourierOp` instead of this operator. It automatically detects if a
            non-uniform or regular fast Fourier transformation is required and can also be constructed automatically
            from a `mrpro.data.KData` object.

        .. note::
            The NUFFT is scaled such that it matches 'orthonormal' FFT scaling for cartesian trajectories.
            This is different from other packages, which apply scaling based on the size of the oversampled grid.

        Parameters
        ----------
        direction
            direction along which non-uniform FFT is applied
        recon_matrix
            Dimension of the reconstructed image. If this is `~mrpro.data.SpatialDimension` only values of directions
            will be used. Otherwise, it should be a `Sequence` of the same length as direction.
        encoding_matrix
            Dimension of the encoded k-space. If this is `~mrpro.data.SpatialDimension` only values of directions will
            be used. Otherwise, it should be a `Sequence` of the same length as direction.
        traj
            The k-space trajectories where the frequencies are sampled.
        oversampling
            Oversampling used for interpolation in non-uniform FFTs.
            On GPU, 2.0 uses an optimized kernel, any value > 1.0 will work.
            On CPU, there are kernels for 2.0 and 1.25. The latter saves memory. Set to 0.0 for automatic selection.
        """
        super().__init__()

        # Convert to negative indexing
        direction_dict = {'z': -3, 'y': -2, 'x': -1, -3: -3, -2: -2, -1: -1}
        self._direction_zyx = tuple(direction_dict[d] for d in direction)
        # Directions are assumed to be in the order -3, -2, -1
        direction_index = np.argsort(self._direction_zyx)
        self._direction_zyx = tuple(self._direction_zyx[i] for i in direction_index)
        if len(direction) != len(set(self._direction_zyx)):
            raise ValueError(f'Directions must be unique. Normalized directions are {self._direction_zyx}')
        if not self._direction_zyx:
            return
        if len(self._direction_zyx) != 1 and len(self._direction_zyx) != 2 and len(self._direction_zyx) != 3:
            raise ValueError('Only 0D, 1D, 2D or 3D NUFFT is supported')

        trajectory = [
            ks for ks, i in zip((traj.kz, traj.ky, traj.kx), (-3, -2, -1), strict=True) if i in self._direction_zyx
        ]

        # Find out along which dimensions (k0, k1 or k2) nufft needs to be applied, i.e. where it is not singleton
        dimension_210: list[int] = []
        for dim in (-3, -2, -1):
            for ks in trajectory:
                if ks.shape[dim] > 1:
                    dimension_210.append(dim)
                    break  # one case where nufft is needed is enough for each dimension

        # For e.g. single shot acquisitions the number of dimensions do not necessarily match the number of
        # directions. This leads to a mismatch between reconstructed and expected dimensions. To avoid this we try
        # to find the most logical solution, i.e. add another singleton direction
        if len(self._direction_zyx) > len(dimension_210):
            for dim in (-1, -2, -3):
                if dim not in dimension_210 and all(ks.shape[dim] == 1 for ks in (traj.kz, traj.ky, traj.kx)):
                    dimension_210.append(dim)
                if len(self._direction_zyx) == len(dimension_210):
                    break
            dimension_210.sort()
        self._dimension_210 = tuple(dimension_210)

        if len(self._direction_zyx) != len(self._dimension_210):
            raise ValueError(
                f'Mismatch between number of nufft directions {self._direction_zyx} and dims {dimension_210}'
            )

        if isinstance(recon_matrix, SpatialDimension):
            im_size = tuple([recon_matrix.zyx[d] for d in self._direction_zyx])
        else:
            if (n_recon_matrix := len(recon_matrix)) != (n_nufft_dir := len(self._direction_zyx)):
                raise ValueError(f'recon_matrix should have {n_nufft_dir} entries but has {n_recon_matrix}')
            im_size = tuple(recon_matrix[i] for i in direction_index)
        assert len(im_size) == 1 or len(im_size) == 2 or len(im_size) == 3  # mypy  # noqa: S101

        if isinstance(encoding_matrix, SpatialDimension):
            k_size = tuple([int(encoding_matrix.zyx[d]) for d in self._direction_zyx])
        else:
            if (n_enc_matrix := len(encoding_matrix)) != (n_nufft_dir := len(self._direction_zyx)):
                raise ValueError(f'encoding_matrix should have {n_nufft_dir} entries but has {n_enc_matrix}')
            k_size = tuple(encoding_matrix[i] for i in direction_index)

        omega_list = [
            k * 2 * torch.pi / ks
            for k, ks in zip(
                trajectory,
                k_size,
                strict=True,
            )
        ]
        # Broadcast shapes not always needed but also does not hurt
        omega_list = list(torch.broadcast_tensors(*omega_list))
        omega = torch.concatenate(omega_list, dim=-4)  # use the 'coil' dim for the direction
        self._traj_broadcast_shape = omega.shape
        keep_dims_210 = [-4, *dimension_210]  # -4 is always coil
        permute_210 = [i for i in range(-omega.ndim, 0) if i not in keep_dims_210] + keep_dims_210
        # omega should be (sep_dims, 1, 2 or 3, nufft_dimensions)
        omega = omega.permute(*permute_210)
        omega = omega.flatten(end_dim=-len(keep_dims_210) - 1).flatten(start_dim=-len(keep_dims_210) + 1)
        # scaling independent of nufft oversampling, matches FFT scaling for cartesian trajectories
        self.scale = torch.tensor(k_size).prod().sqrt().reciprocal()
        self.oversampling = oversampling

        # we want to rearrange everything into (sep_dims)(joint_dims)(nufft_dims) where sep_dims are dimension
        # where the traj changes, joint_dims are dimensions where the traj does not change and nufft_dims are the
        # dimensions along which the nufft is applied. We have to do this for the (z-y-x) and (k2-k1-k0) space
        # separately. If we know two of the three dimensions we can infer the rest. We cannot do the other
        # dimensions here because they might be different between data and trajectory.
        self._joint_dims_210 = [
            d for d in [-3, -2, -1] if d not in dimension_210 and self._traj_broadcast_shape[d] == 1
        ]
        self._joint_dims_210.append(-4)  # -4 is always coil and always a joint dimension

        traj_shape = torch.as_tensor([k.shape[-3:] for k in (traj.kz, traj.ky, traj.kx)])
        self._joint_dims_zyx = []
        for dzyx in [-3, -2, -1]:
            if dzyx not in self._direction_zyx:
                dim210_non_singleton = [d210 for d210 in [-3, -2, -1] if traj_shape[dzyx, d210] > 1]
                if all(all(traj_shape[self._direction_zyx, d] == 1) for d in dim210_non_singleton):
                    self._joint_dims_zyx.append(dzyx)
        self._joint_dims_zyx.append(-4)  # -4 is always coil and always a joint dimension

        self._im_size = im_size
        self._omega = omega.contiguous()

    def _separate_joint_dimensions(
        self, data_ndim: int
    ) -> tuple[Sequence[int], Sequence[int], Sequence[int], Sequence[int]]:
        """Get the separate and joint dimensions for the current data.

        Parameters
        ----------
        data_ndim
            number of dimensions of the data

        Returns
        -------
            separate dimensions along zyx,
            permutation rule for zyx,
            separate dimensions along 210,
            permutation rule for 210.

        """
        # We did most in _init_ and here we only have to check the other dimensions.
        joint_dims_other = []
        for d in range(-data_ndim, -4):
            if abs(d) > len(self._traj_broadcast_shape) or self._traj_broadcast_shape[d] == 1:
                joint_dims_other.append(d)

        sep_dims_zyx = [
            d for d in range(-data_ndim, 0) if d not in [*joint_dims_other, *self._joint_dims_zyx, *self._direction_zyx]
        ]
        permute_zyx = [*sep_dims_zyx, *joint_dims_other, *self._joint_dims_zyx, *self._direction_zyx]
        sep_dims_210 = [
            d for d in range(-data_ndim, 0) if d not in [*joint_dims_other, *self._joint_dims_210, *self._dimension_210]
        ]
        permute_210 = [*sep_dims_210, *joint_dims_other, *self._joint_dims_210, *self._dimension_210]
        return sep_dims_zyx, permute_zyx, sep_dims_210, permute_210

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply Non-Uniform Fast Fourier Transform (NUFFT) from image to k-space (Type 2).

        Transforms data from image space to non-uniform k-space locations.

        Parameters
        ----------
        x
            Image-space data with selected spatial dimensions as trailing axes, e.g.
            `(..., coils, z, y, x)` for 3D or `(..., coils, y, x)` for 2D.

        Returns
        -------
            K-space data at the non-uniform trajectory locations with the same leading
            dimensions as the input and trailing sampled dimensions, e.g.
            `(..., coils, k2, k1, k0)` for 3D or `(..., coils, k1, k0)` for 2D.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of NonUniformFastFourierOp.

        .. note::
            Prefer calling the instance of the NonUniformFastFourierOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if len(self._direction_zyx):
            if x.device.type == 'cpu' and self.oversampling not in (0.0, 1.25, 2.0):
                raise ValueError('Only oversampling 1.25 and 2.0 are supported on CPU')
            elif x.device.type not in ('cuda', 'cpu'):
                raise ValueError('Only CPU and CUDA are supported')
            # We rearrange x into (sep_dims, joint_dims, nufft_directions)
            sep_dims_zyx, permute_zyx, _, permute_210 = self._separate_joint_dimensions(x.ndim)
            unpermute_210 = torch.tensor(permute_210).argsort().tolist()

            x = x.permute(*permute_zyx)
            unflatten_shape = x.shape[: -len(self._direction_zyx)]
            # combine sep_dims
            x = x.flatten(end_dim=len(sep_dims_zyx) - 1) if len(sep_dims_zyx) else x[None, :]
            # combine joint_dims
            x = x.flatten(start_dim=1, end_dim=-len(self._direction_zyx) - 1)
            # cufinufft needs c contiguous, otherwise it warns.
            x = x.contiguous()
            x = torch.vmap(partial(finufft_type2, upsampfac=self.oversampling, modeord=0, isign=-1))(self._omega, x)
            x = x * self.scale
            shape_210 = [self._traj_broadcast_shape[i] for i in self._dimension_210]
            x = x.reshape(*unflatten_shape, *shape_210)
            x = x.permute(*unpermute_210)
        return (x,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply adjoint NUFFT from k-space to image space (Type 1).

        Transforms data from non-uniform k-space to a Cartesian image grid.

        Parameters
        ----------
        x
            Coil k-space data at non-uniform locations,
            with shape `(..., coils, k2, k1, k0)`.

        Returns
        -------
            Coil image data on a Cartesian grid,
            with shape `(..., coils, z, y, x)`.
        """
        if len(self._direction_zyx):
            if x.device.type == 'cpu' and self.oversampling not in (0.0, 1.25, 2.0):
                raise ValueError('Only oversampling 1.25 and 2.0 are supported on CPU')
            elif x.device.type not in ('cuda', 'cpu'):
                raise ValueError('Only CPU and CUDA are supported')
            # We rearrange x into (sep_dims, joint_dims, nufft_directions)
            _, permute_zyx, sep_dims_210, permute_210 = self._separate_joint_dimensions(x.ndim)
            unpermute_zyx = torch.tensor(permute_zyx).argsort().tolist()

            x = x.permute(*permute_210)
            unflatten_other_shape = x.shape[: -len(self._dimension_210) - 1]  # -1 for coil
            # combine sep_dims
            x = x.flatten(end_dim=len(sep_dims_210) - 1) if len(sep_dims_210) else x[None, :]
            # combine joint_dims and nufft_dims
            x = x.flatten(start_dim=1, end_dim=-len(self._dimension_210) - 1).flatten(start_dim=2)
            # cufinufft needs c contiguous, otherwise it warns.
            x = x.contiguous()
            x = torch.vmap(
                partial(finufft_type1, upsampfac=self.oversampling, modeord=0, isign=1, output_shape=self._im_size)
            )(self._omega, x)
            x = x * self.scale

            x = x.reshape(*unflatten_other_shape, -1, *x.shape[-len(self._direction_zyx) :])
            x = x.permute(*unpermute_zyx)
        return (x,)

    def toeplitz(
        self,
        weight: torch.Tensor | DcfData | None = None,
        subspace: torch.Tensor | PCACompressionOp | None = None,
        subspace_dim: int = 0,
    ) -> LinearOperator:
        """Return the Toeplitz Gram operator.

        Parameters
        ----------
        weight
            Optional density compensation or k-space weights. If provided, calculates
            the normal operator corresponding to :math:`F^H W F`.
        subspace
            Optional temporal subspace basis of shape `(n_timepoints, n_coefficients)`, or
            a `PCACompressionOp` whose adjoint defines the temporal expansion basis.
        subspace_dim
            Dimension of the coefficient channel in the input/output tensors when `subspace` is provided.
        """
        if subspace is None:
            return NonUniformFastFourierOpGramOp(self, weight=weight)
        return SubspaceNonUniformFastFourierOpGramOp(
            self, subspace_basis=subspace, weight=weight, subspace_dim=subspace_dim
        )

    @property
    def gram(self) -> LinearOperator:
        """Return the unweighted Toeplitz Gram operator."""
        return self.toeplitz()

    def __repr__(self) -> str:
        """Representation method for NUFFT operator."""
        device = self._omega.device if self._omega is not None else 'none'
        zyx = ['z', 'y', 'x']
        k2k1k0 = ['k2', 'k1', 'k0']
        direction_zyx = tuple(zyx[i] for i in self._direction_zyx if i in range(-3, 0))
        dimension_210 = tuple(k2k1k0[i] for i in self._dimension_210 if i in range(-3, 0))
        recon_size_str = ', '.join(f'{dim}={size}' for dim, size in zip(direction_zyx, self._im_size, strict=False))
        direction_zyx_str = ', '.join(direction_zyx)
        dimension_210_str = ', '.join(dimension_210)
        out = (
            f'{type(self).__name__} on device: {device}\n'
            f'Dimension(s) along which NUFFT is applied: ({direction_zyx_str}) / ({dimension_210_str})\n'
            f'Reconstructed image size {recon_size_str}'
        )
        return out


def symmetrize(kernel: torch.Tensor, rank: int) -> torch.Tensor:
    """Enforce hermitian symmetry on the kernel. Returns only half of the kernel."""
    flipped = kernel
    for d in range(-rank, 0):
        flipped = flipped.index_select(d, -1 * torch.arange(flipped.shape[d], device=flipped.device) % flipped.size(d))
    kernel = (kernel + flipped.conj()) / 2
    last_len = kernel.shape[-1]
    return kernel[..., : last_len // 2 + 1]


def gram_nufft_kernel(
    weight: torch.Tensor, trajectory: torch.Tensor, recon_shape: tuple[int] | tuple[int, int] | tuple[int, int, int]
) -> torch.Tensor:
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
        Real valued convolution kernel for `~mrpro.operator.NonUniformFastFourierOpGramOp`, already in Fourier
        space.
    """
    rank = trajectory.shape[-2]
    if weight.dtype.is_complex:
        raise ValueError('Only real weights are supported')
    # Instead of doing one adjoint nufft with double the recon size in all dimensions,
    # we do two adjoint nuffts per dimensions, saving a lot of memory.
    adjnufft = torch.vmap(partial(finufft_type1, modeord=0, isign=1, output_shape=recon_shape))

    kernel = torch.zeros(
        *weight.shape[:2], *[r * 2 for r in recon_shape], dtype=weight.dtype.to_complex(), device=weight.device
    )
    shifts = (torch.tensor(recon_shape) / 2).unsqueeze(-1).to(trajectory)
    for flips in list(product([1, -1], repeat=rank)):
        flipped_trajectory = trajectory * torch.tensor(flips).to(trajectory).unsqueeze(-1)
        kernel_part = adjnufft(flipped_trajectory, torch.polar(weight, (shifts * flipped_trajectory).sum(-2, True)))
        idx: list[slice | EllipsisType] = [...]  # which part of the kernel to is currently being processed
        for dim, flip in zip(range(-rank, 0), flips, strict=True):
            if flip > 0:  # first half in the dimension
                idx.append(slice(0, kernel_part.size(dim)))
            else:  # second half in the dimension
                idx.append(slice(kernel_part.size(dim) + 1, None))
                kernel_part = kernel_part.index_select(
                    dim,
                    torch.arange(kernel_part.size(dim) - 1, 0, -1, device=kernel.device),
                )  # flip
        kernel[tuple(idx)] = kernel_part

    kernel = symmetrize(kernel, rank)
    kernel = torch.fft.hfftn(kernel, dim=list(range(-rank, 0)), norm='backward')
    return kernel


class NonUniformFastFourierOpGramOp(LinearOperator, adjoint_as_backward=True):
    """Gram operator for `NonUniformFastFourierOp`.

    Implements the adjoint of the forward operator of the non-uniform Fast Fourier operator, i.e. the gram operator
    `NUFFT.H@NUFFT`.

    Uses a convolution, implemented as multiplication in Fourier space, to calculate the gram operator
    for the Toeplitz NUFFT operator.

    This should not be used directly, but rather through the `~NonUniformFastFourierOp.gram` method of a
    `NonUniformFastFourierOp` object.

    .. note::
        Consider calling .half() on the operator to save memory at the cost of precision.
    """

    _kernel: torch.Tensor | None

    def __init__(self, nufft_op: NonUniformFastFourierOp, weight: torch.Tensor | DcfData | None = None) -> None:
        """Initialize the gram operator.

        Parameters
        ----------
        nufft_op
            The py:class:`NonUniformFastFourierOp` to calculate the gram operator for.
        weight
            Optional density compensation weights. If provided, calculates F^H W F.
        """
        super().__init__()
        if not nufft_op._dimension_210:
            self._kernel = None
            return

        self._dim = nufft_op._direction_zyx
        self._recon_shape = nufft_op._im_size

        if isinstance(weight, DcfData):
            weight = weight.data
        weight_shape = [*nufft_op._traj_broadcast_shape[:-4], 1, *nufft_op._traj_broadcast_shape[-3:]]
        if weight is None:
            weight = torch.ones(weight_shape, device=nufft_op._omega.device, dtype=nufft_op._omega.dtype.to_real())
        else:
            weight = weight.to(nufft_op._omega.device).broadcast_to(weight_shape)
        # We rearrange weight into (sep_dims, joint_dims, nufft_directions)
        _, permute_zyx, sep_dims_210, permute_210 = nufft_op._separate_joint_dimensions(weight.ndim)
        unpermute_zyx = torch.tensor(permute_zyx).argsort().tolist()
        weight = weight.permute(*permute_210)
        unflatten_other_shape = weight.shape[: -len(nufft_op._dimension_210) - 1]  # -1 for coil
        if sep_dims_210:  # combine sep_dims
            weight = weight.flatten(end_dim=len(sep_dims_210) - 1)
        else:
            weight = weight[None, :]
        # combine joint_dims and nufft_dims
        weight = weight.flatten(start_dim=1, end_dim=-len(nufft_op._dimension_210) - 1).flatten(start_dim=2)

        kernel = gram_nufft_kernel(weight, nufft_op._omega, nufft_op._im_size)
        kernel = kernel.reshape(*unflatten_other_shape, -1, *kernel.shape[-len(nufft_op._direction_zyx) :])
        self._kernel = kernel.permute(*unpermute_zyx) * nufft_op.scale.item() ** 2

    @property
    def recon_matrix(self) -> SpatialDimension[int]:
        """Expected reconstructed image shape as a spatial dimension."""
        zyx = [1, 1, 1]
        for direction, size in zip(self._dim, self._recon_shape, strict=True):
            zyx[direction] = size
        return SpatialDimension(*zyx)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of NonUniformFastFourierOpGramOp.

        .. note::
            Prefer calling the instance of the NonUniformFastFourierOpGramOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        # We do the fft and cropping here directly, as it faster and more memory efficient
        # then using the operators. As this will be a bottleneck in iterative 3D reconstructions,
        # it is worth the specialization.

        # This function on its own is also not autograd save (in-place ops), but we rely on the
        # adjoint-as-backward trick for linear operators to make it work

        if self._kernel is None:
            return (x,)

        padded_shape = [2 * s for s in self._recon_shape]
        spatial_crop: list[slice | EllipsisType] = [..., slice(None), slice(None), slice(None)]
        for d, s in zip(self._dim, self._recon_shape, strict=True):
            spatial_crop[d] = slice(0, s)

        x = torch.fft.fftn(x, s=padded_shape, dim=self._dim)
        x.mul_(self._kernel)
        x = torch.fft.ifftn(x, dim=self._dim, out=x)
        x = x[tuple(spatial_crop)]
        out = x.clone()  # clone to deallocate the larger x on exit

        return (out,)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the Gram operator of the NonUniformFastFourierOp (NUFFT.H @ NUFFT).

        This operation applies the composition of the adjoint NUFFT operator
        and the forward NUFFT operator. It is implemented via a convolution
        with a precomputed kernel.

        Parameters
        ----------
        x
            Input tensor, typically image-space data with shape `(..., coils, z, y, x)`.

        Returns
        -------
            Image-space data after applying `NUFFT.H @ NUFFT`, with the same shape as `x`.
        """
        return super().__call__(x)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the Gram operator.

        Since the Gram operator (NUFFT.H @ NUFFT) is self-adjoint,
        this method calls the forward operation.

        Parameters
        ----------
        x
            Input tensor, typically image-space data with shape `(..., coils, z, y, x)`.

        Returns
        -------
            Image-space data after applying the adjoint Gram operator, with the same shape as `x`.
        """
        return super().__call__(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self


class SubspaceNonUniformFastFourierOpGramOp(LinearOperator, adjoint_as_backward=True):
    """Subspace Gram operator for `NonUniformFastFourierOp`.

    This operator acts on temporally compressed coefficient images and applies the
    compressed normal operator corresponding to a time-varying non-uniform Fourier
    encoding. The Toeplitz kernel is precomputed once during initialization.

    The current implementation is intentionally restricted to the common MRF case:
    exactly one varying separate trajectory dimension, interpreted as time, and no
    additional non-singleton joint trajectory dimensions.
    """

    _kernel: torch.Tensor | None

    def __init__(
        self,
        nufft_op: NonUniformFastFourierOp,
        subspace_basis: torch.Tensor | PCACompressionOp,
        weight: torch.Tensor | DcfData | None = None,
        subspace_dim: int = 0,
    ) -> None:
        """Initialize the subspace Gram operator.

        Parameters
        ----------
        nufft_op
            The non-uniform Fourier operator defining the trajectory and spatial geometry.
        subspace_basis
            Temporal subspace basis of shape `(n_timepoints, n_coefficients)`, or
            a `PCACompressionOp` whose adjoint defines the temporal expansion basis.
        weight
            Optional density compensation or k-space weights. If provided, calculates
            the compressed normal operator corresponding to :math:`F^H W F`.
        subspace_dim
            Dimension of the coefficient channel in the input/output tensors.
        """
        super().__init__()
        if not nufft_op._dimension_210:
            self._kernel = None
            return

        self._dim = nufft_op._direction_zyx
        self._recon_shape = nufft_op._im_size
        self._subspace_dim = subspace_dim

        if isinstance(subspace_basis, PCACompressionOp):
            basis = subspace_basis.compression_matrix.mH
        else:
            basis = subspace_basis
        if basis.ndim > 2:
            basis = basis.squeeze(tuple(range(basis.ndim - 2)))
        if basis.ndim > 2:
            raise ValueError(f'Basis cannot contain non-singleton batch dimensions; got shape {basis.shape}.')
        if basis.ndim == 1:  # rank-1 special case: only one coefficient
            basis = basis.unsqueeze(-1)
        basis = basis.to(device=nufft_op._omega.device)

        if isinstance(weight, DcfData):
            weight = weight.data
        weight_shape = [*nufft_op._traj_broadcast_shape[:-4], 1, *nufft_op._traj_broadcast_shape[-3:]]
        if weight is None:
            weight = torch.ones(weight_shape, device=nufft_op._omega.device, dtype=nufft_op._omega.dtype.to_real())
        else:
            weight = weight.to(nufft_op._omega.device).broadcast_to(weight_shape)
        _, _permute_zyx, sep_dims_210, permute_210 = nufft_op._separate_joint_dimensions(weight.ndim)
        weight = weight.permute(*permute_210)
        if sep_dims_210:
            weight = weight.flatten(end_dim=len(sep_dims_210) - 1)
        else:
            weight = weight[None, :]
        weight = weight.flatten(start_dim=1, end_dim=-len(nufft_op._dimension_210) - 1).flatten(start_dim=2)

        omega = nufft_op._omega

        if len(sep_dims_210) > 1:
            raise NotImplementedError(
                'SubspaceNonUniformFastFourierOpGramOp currently only supports a single varying separate dimension.'
            )
        if basis.shape[0] != (n_timepoints := weight.shape[0]):
            raise ValueError(
                f'subspace_basis has {basis.shape[0]} time points, but the trajectory varies over {n_timepoints}.'
            )

        subspace_kernel: torch.Tensor | None = None
        for basis_row, weight_row, omega_row in zip(basis, weight, omega, strict=True):
            current = gram_nufft_kernel(weight_row[None], omega_row[None], nufft_op._im_size)[0, 0]
            coeff_outer = basis_row.conj()[:, None] * basis_row[None, :]
            term = unsqueeze_right(coeff_outer, current.ndim) * current
            subspace_kernel = term if subspace_kernel is None else subspace_kernel + term

        assert subspace_kernel is not None  # noqa: S101
        kernel_shape = [1, 1, 1]
        for direction, size in zip(self._dim, subspace_kernel.shape[2:], strict=True):
            kernel_shape[direction] = size
        subspace_kernel = subspace_kernel.reshape(*subspace_kernel.shape[:2], *kernel_shape)
        self._kernel = subspace_kernel * nufft_op.scale.item() ** 2

    @property
    def n_coefficients(self) -> int:
        """Number of compressed coefficient channels."""
        if self._kernel is None:
            raise RuntimeError('n_coefficients is undefined if there are no NUFFT axes.')
        return self._kernel.shape[0]

    @property
    def recon_matrix(self) -> SpatialDimension[int]:
        """Expected reconstructed image shape as a spatial dimension."""
        zyx = [1, 1, 1]
        for direction, size in zip(self._dim, self._recon_shape, strict=True):
            zyx[direction] = size
        return SpatialDimension(*zyx)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of SubspaceNonUniformFastFourierOpGramOp.

        .. note::
            Prefer calling the instance of the SubspaceNonUniformFastFourierOpGramOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        # We do the fft, coefficient mixing, and cropping here directly, as it is faster and more memory efficient
        # than using separate operators.

        # This function on its own is also not autograd save (in-place ops), but we rely on the
        # adjoint-as-backward trick for linear operators to make it work.
        if self._kernel is None:
            return (x,)
        subspace_dim = normalize_index(x.ndim, self._subspace_dim)
        if x.shape[subspace_dim] != self.n_coefficients:
            raise ValueError(
                f'Input has {x.shape[self._subspace_dim]} coefficients along subspace_dim={self._subspace_dim}, '
                f'expected {self.n_coefficients}.'
            )
        x = x.movedim(subspace_dim, 0)

        padded_shape = [2 * s for s in self._recon_shape]
        spatial_crop: list[slice | EllipsisType] = [..., slice(None), slice(None), slice(None)]
        for d, s in zip(self._dim, self._recon_shape, strict=True):
            spatial_crop[d] = slice(0, s)

        x = torch.fft.fftn(x, s=padded_shape, dim=self._dim)
        x = einops.einsum(self._kernel.to(x.dtype), x, 'coeff_out coeff_in ..., coeff_in ... -> coeff_out ...')
        x = torch.fft.ifftn(x, dim=self._dim)
        x = x[tuple(spatial_crop)]
        out = x.clone().movedim(0, subspace_dim)
        return (out,)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the compressed Toeplitz Gram operator.

        Parameters
        ----------
        x
            Subspace coefficient images. The coefficient axis is given by `subspace_dim`

        Returns
        -------
            Subspace coefficient images after applying the compressed Gram operator,
            with the same shape as `x`.
        """
        return super().__call__(x)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the compressed Toeplitz Gram operator.

        Since the compressed Gram operator is self-adjoint, this method calls the
        forward operation.

        Parameters
        ----------
        x
            Subspace coefficient images. The coefficient axis is given by `subspace_dim`


        Returns
        -------
            Subspace coefficient images after applying the adjoint compressed Gram operator,
            with the same shape as `x`.
        """
        return super().__call__(x)

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint operator of the gram operator."""
        return self
