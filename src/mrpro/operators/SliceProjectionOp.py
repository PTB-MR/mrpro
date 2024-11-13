"""Class for 3D->2D Projection Operator."""

import itertools
import warnings
from collections.abc import Callable, Sequence
from typing import Literal, TypeAlias

import einops
import numpy as np
import torch
from numpy._typing import _NestedSequence as NestedSequence
from torch import Tensor

from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.utils.slice_profiles import SliceSmoothedRectangular


class _MatrixMultiplicationCtx(torch.autograd.function.FunctionCtx):
    """Autograd context for matrix multiplication, used for type hinting."""

    x_is_complex: bool
    saved_tensors: tuple[Tensor]


class _MatrixMultiplication(torch.autograd.Function):
    """Helper for matrix multiplication.

    For sparse matrices, it can be more efficient to have a
    separate representation of the adjoint matrix to be used
    in the backward pass.

    """

    @staticmethod
    def forward(x: Tensor, matrix: Tensor, matrix_adjoint: Tensor) -> Tensor:  # noqa: ARG004
        if x.is_complex() == matrix.is_complex():
            return matrix @ x
        # required for sparse matrices to support mixed complex/real multiplication
        elif x.is_complex():
            return torch.complex(matrix @ x.real, matrix @ x.imag)
        else:
            return torch.complex(matrix.real @ x, matrix.imag @ x)

    @staticmethod
    def setup_context(
        ctx: _MatrixMultiplicationCtx,
        inputs: tuple[Tensor, Tensor, Tensor],
        outputs: tuple[Tensor],  # noqa: ARG004
    ) -> None:
        x, _, matrix_adjoint = inputs
        ctx.x_is_complex = x.is_complex()
        ctx.save_for_backward(matrix_adjoint)

    @staticmethod
    def backward(ctx: _MatrixMultiplicationCtx, *grad_output: Tensor) -> tuple[Tensor, None, None]:
        (matrix_adjoint,) = ctx.saved_tensors
        if ctx.x_is_complex:
            if matrix_adjoint.is_complex() == grad_output[0].is_complex():
                grad_x = matrix_adjoint @ grad_output[0]
            elif matrix_adjoint.is_complex():
                grad_x = torch.complex(matrix_adjoint.real @ grad_output[0], matrix_adjoint.imag @ grad_output[0])
            else:
                grad_x = torch.complex(matrix_adjoint @ grad_output[0].real, matrix_adjoint @ grad_output[0].imag)
        else:  # real grad
            grad_x = matrix_adjoint.real @ grad_output[0].real
            if matrix_adjoint.is_complex() and grad_output[0].is_complex():
                grad_x -= matrix_adjoint.imag @ grad_output[0].imag
        return grad_x, None, None


TensorFunction: TypeAlias = Callable[[Tensor], Tensor]


class SliceProjectionOp(LinearOperator):
    """Slice Projection Operator.

    This operation samples from a 3D Volume a slice with a given rotation and shift
    (relative to the center of the volume) according to the slice_profile.
    It can, for example, be used to describe the slice selection of a 2D MRI sequence
    from the 3D Volume.

    The projection will be done by sparse matrix multiplication.

    Rotation, shift, and profile can have (multiple) batch dimensions. These dimensions will
    be broadcasted to a common shape and added to the front of the volume.
    Different settings for different volume batches are NOT supported, consider creating multiple
    operators if required.
    """

    matrix: Tensor | None
    matrix_adjoint: Tensor | None

    def __init__(
        self,
        input_shape: SpatialDimension[int],
        slice_rotation: Rotation | None = None,
        slice_shift: float | Tensor = 0.0,
        slice_profile: TensorFunction | np.ndarray | NestedSequence[TensorFunction] | float = 2.0,
        optimize_for: Literal['forward', 'adjoint', 'both'] = 'both',
    ):
        """Create a module that represents the 'projection' of a volume onto a plane.

        This operation samples from a 3D Volume a slice with a given rotation and shift
        (relative to the center of the volume) according to the slice_profile.
        It can, for example, be used to describe the slice selection of a 2D MRI sequence
        from the 3D Volume.

        The projection will be done by sparse matrix multiplication.

        Rotation, shift, and profile can have (multiple) batch dimensions. These dimensions will
        be broadcasted to a common shape and added to the front of the volume.
        Different settings for different volume batches are NOT supported, consider creating multiple
        operators if required.


        Parameters
        ----------
        input_shape
            Shape of the 3D volume to sample from (z, y, x)
        slice_rotation
            Rotation that describes the orientation of the plane. If None,
            an identity rotation is used.
        slice_shift
            Offset of the plane in the volume perpendicular plane from the center of the volume.
            (The center of a 4 pixel volume is between 1 and 2.)
        slice_profile
            A function returning the relative intensity of the slice profile at a position x
            (relative to the nominal profile center). This can also be a nested Sequence or an
            numpy array of functions.
            If it is a single float, it will be interpreted as the FWHM of a rectangular profile.
        optimize_for
            Whether to optimize for forward or adjoint operation or both.
            Optimizing for both takes more memory but is faster for both operations.

        """
        super().__init__()
        if isinstance(slice_profile, float | int):
            slice_profile = SliceSmoothedRectangular(slice_profile, 0.0)
        slice_profile_array = np.array(slice_profile)

        if slice_rotation is None:
            slice_rotation = Rotation.identity()

        max_shape = max(input_shape.z, input_shape.y, input_shape.x)

        def _find_width(slice_profile: TensorFunction) -> int:
            # figure out how far along the profile we have to consider values
            # clip up to 0.01 of intensity on both sides
            test_values = torch.arange(-max_shape, max_shape, max_shape)
            profile = slice_profile(test_values)
            cdf = torch.cumsum(profile, -1) / profile.sum()
            left = test_values[np.argmax(cdf > 0.01)]
            right = test_values[np.argmax(cdf > 0.99)]
            return int(max(left.abs().item(), right.abs().item())) + 1

        widths = np.vectorize(_find_width)(slice_profile_array)

        def _at_least_width_1(slice_profile: TensorFunction):
            test_values = torch.linspace(-0.5, 0.5, 100)
            return (slice_profile(test_values) > 1e-6).all()

        if not np.vectorize(_at_least_width_1)(slice_profile_array).all():
            raise ValueError(
                'The slice profile must have a width of at least 1 voxel,'
                ' i.e. the profile should be greater then 1e-6 in (-0.5,0.5)'
            )

        slice_shift_tensor = torch.atleast_1d(torch.as_tensor(slice_shift))
        batch_shapes = torch.broadcast_shapes(slice_rotation.shape, slice_shift_tensor.shape, slice_profile_array.shape)
        rotation_quats = torch.broadcast_to(slice_rotation.as_quat(), (*batch_shapes, 4)).reshape(-1, 4)
        slice_rotation = Rotation(rotation_quats, normalize=False, copy=False)
        slice_shift_tensor = torch.broadcast_to(slice_shift_tensor, batch_shapes).flatten()
        slice_profile_array = np.broadcast_to(slice_profile_array, batch_shapes).ravel()
        widths = np.broadcast_to(widths, batch_shapes).ravel()

        matrices = [
            SliceProjectionOp.projection_matrix(
                input_shape,
                SpatialDimension(1, max_shape, max_shape),
                offset=torch.tensor([shift, 0.0, 0.0]),
                slice_function=f,
                rotation=rot,
                w=int(w),
            )
            for rot, shift, f, w in zip(slice_rotation, slice_shift_tensor, slice_profile_array, widths, strict=True)
        ]
        matrix = SliceProjectionOp.join_matrices(matrices)

        # in csr format the matmul is faster, but saving one for forward and adjoint takes more memory
        with warnings.catch_warnings():
            # beta status in pytorch causes a warning to be printed
            warnings.filterwarnings('ignore', category=UserWarning, message='Sparse')
            if optimize_for == 'forward':
                self.register_buffer('matrix', matrix.to_sparse_csr())
                self.matrix_adjoint = None
            elif optimize_for == 'adjoint':
                self.register_buffer('matrix_adjoint', matrix.H.to_sparse_csr())
                self.matrix = None
            elif optimize_for == 'both':
                self.register_buffer('matrix_adjoint', matrix.H.to_sparse_csr())
                self.register_buffer('matrix', matrix.to_sparse_csr())

            else:
                raise ValueError("optimize_for must be one of 'forward', 'adjoint', 'both'")

        self._range_shape: tuple[int] = (*batch_shapes, 1, max_shape, max_shape)
        self._domain_shape = input_shape.zyx

    def forward(self, x: Tensor) -> tuple[Tensor]:
        """Transform from a 3D Volume to a 2D Slice.

        Parameters
        ----------
        x
            3D Volume with shape (..., z, y, x)
            with z, y, x matching the input_shape

        Returns
        -------
        A 2D slice with shape (..., 1, max(z, y, x), (max(z, y, x)))
        """
        match (self.matrix, self.matrix_adjoint):
            # selection based on the optimize_for setting
            case (None, None):
                raise RuntimeError('Either matrix or matrix adjoint must be set')
            case (matrix, None) if matrix is not None:
                matrix_adjoint = matrix.H
            case (None, matrix_adjoint) if matrix_adjoint is not None:
                matrix = matrix_adjoint.H
            case (matrix, matrix_adjoint):
                ...

        # For the (unusual case) of batched volumes, we will apply for each element in series
        xflat = torch.atleast_2d(einops.rearrange(x, '... x y z -> (...) (x y z)'))
        yl = [_MatrixMultiplication.apply(x, matrix, matrix_adjoint) for x in xflat]

        y = torch.stack([el.reshape(self._range_shape) for el in yl], -4)
        y = y.reshape(*y.shape[:-4], *x.shape[:-3], *y.shape[-3:])
        return (y,)

    def adjoint(self, x: Tensor) -> tuple[Tensor,]:
        """Transform from a 2D slice to a 3D Volume.

        Parameters
        ----------
        x
            2D Slice with shape (..., 1, max(z, y, x), (max(z, y, x)))
            with z, y, x matching the input_shape

        Returns
        -------
        A 3D Volume with shape (..., z, y, x)
           with z, y, x matching the input_shape
        """
        match (self.matrix, self.matrix_adjoint):
            # selection based on the optimize_for setting
            case (None, None):
                raise RuntimeError('Either matrix or matrix adjoint must be set')
            case (matrix, None) if matrix is not None:
                matrix_adjoint = matrix.H
            case (None, matrix_adjoint) if matrix_adjoint is not None:
                matrix = matrix_adjoint.H
            case (matrix, matrix_adjoint):
                ...

        # For the (unusual case) of batched volumes, we will apply for each element in series
        n_batchdim = len(self._range_shape) - 3
        # x_domainbatch_range has all volume batch dimensions moved to the front
        x_domainbatch_range = x.moveaxis(tuple(range(n_batchdim, x.ndim - 3)), tuple(range(x.ndim - 3 - n_batchdim)))
        # x_flatdomainbatch_flatrange is 2D with shape
        # (all batch dimensions of the volume flattened, all range dimensions flattened)
        x_domainbatch_flatrange = torch.atleast_2d(x_domainbatch_range.flatten(start_dim=-len(self._range_shape)))
        x_flatdomainbatch_flatrange = x_domainbatch_flatrange.flatten(end_dim=-2)
        # y_flatdomainbatch has shape (all batch dimensions of the volume flattened, *range dimensions)
        y_flatdomainbatch = torch.stack(
            [
                _MatrixMultiplication.apply(x, matrix_adjoint, matrix).reshape(self._domain_shape)
                for x in x_flatdomainbatch_flatrange
            ]
        )
        y = y_flatdomainbatch.reshape(*x.shape[n_batchdim:-3], *y_flatdomainbatch.shape[1:])
        return (y,)

    @staticmethod
    def join_matrices(matrices: Sequence[Tensor]) -> Tensor:
        """Join multiple sparse matrices into a block diagonal matrix.

        Parameters
        ----------
        matrices
            List of sparse matrices to join by stacking them as a block diagonal matrix
        """
        values = []
        target = []
        source = []
        for i, m in enumerate(matrices):
            if not m.shape == matrices[0].shape:
                raise ValueError('all matrices should have the same shape')
            c = m.coalesce()  # we want unique indices
            (ctarget, csource) = c.indices()
            values.append(c.values())
            source.append(csource)
            ctarget = ctarget + i * m.shape[0]
            target.append(ctarget)

        with warnings.catch_warnings():
            # beta status in pytorch causes a warning to be printed
            warnings.filterwarnings('ignore', category=UserWarning, message='Sparse')
            matrix = torch.sparse_coo_tensor(
                indices=torch.stack([torch.cat(target), torch.cat(source)]),
                values=torch.cat(values),
                dtype=torch.float32,
                size=(len(matrices) * m.shape[0], m.shape[1]),
            )
        return matrix

    @staticmethod
    def projection_matrix(
        input_shape: SpatialDimension[int],
        output_shape: SpatialDimension[int],
        rotation: Rotation,
        offset: Tensor,
        w: int,
        slice_function: TensorFunction,
        rotation_center: Tensor | None = None,
    ) -> Tensor:
        """Create a sparse matrix that represents the projection of a volume onto a plane.

        Outside the volume values are approximately zero padded

        Parameters
        ----------
        input_shape
            Shape of the volume to sample from
        output_shape
            Shape of the resulting plane, 2D. Only the x and y values are used.
        rotation
            Rotation that describes the orientation of the plane
        offset: Tensor
            Shift of the plane from the center of the volume in the rotated coordinate system
            in units of the 3D volume, order z, y, x
        w: int
            Factor that determines the number of pixels that are considered in the projection along
            the slice profile direction.
        slice_function
            Function that describes the slice profile.
        rotation_center
            Center of rotation, if None the center of the volume is used,
            i.e. for 4 pixels 0 1 2 3 it is between 1 and 2

        Returns
        -------
        torch.sparse_coo_matrix
            Sparse matrix that represents the projection of the volume onto the plane
        """
        x, y = output_shape.x, output_shape.y

        start_x, start_y = (
            (input_shape.x - x) // 2,
            (input_shape.y - y) // 2,
        )
        pixel_coord_y_x_zyx = torch.stack(
            [
                (input_shape.z / 2 - 0.5) * torch.ones(y, x),  # z coordinates
                *torch.meshgrid(
                    torch.arange(start_y, start_y + y),  # y coordinates
                    torch.arange(start_x, start_x + x),  # x coordinates
                    indexing='ij',
                ),
            ],
            dim=-1,
        )  # coordinates of the 2d output pixels in the coordinate system of the input volume, so shape (y,x,3)
        if offset is not None:
            pixel_coord_y_x_zyx = pixel_coord_y_x_zyx + offset
        if rotation_center is None:
            # default rotation center is the center of the volume, i.e. for 4 pixels
            # 0 1 2 3 it is between 0 and 1
            rotation_center = torch.tensor([input_shape.z / 2 - 0.5, input_shape.y / 2 - 0.5, input_shape.x / 2 - 0.5])
        pixel_rotated_y_x_zyx = rotation(pixel_coord_y_x_zyx - rotation_center) + rotation_center

        # We cast a ray from the pixel normal to the plane in both directions
        # points in the original volume further away then w will not be considered
        ray = rotation(
            torch.stack(
                [
                    torch.arange(-w, w + 1),  # z
                    torch.zeros(2 * w + 1),  # y
                    torch.zeros(2 * w + 1),  # x
                ],
                dim=-1,
            )
        )
        # In all possible directions for each point along the line we consider the eight neighboring points
        # by adding all possible combinations of 0 and 1 to the point and flooring
        offsets = torch.tensor(list(itertools.product([0, 1], repeat=3)))
        # all points that influence a pixel
        # x,y,8-neighbors,(2*w+1)-raylength,3-dimensions input_shape.xinput_shape.yinput_shape.z)
        points_influencing_pixel = (
            einops.rearrange(pixel_rotated_y_x_zyx, '   y x zyxdim -> y x 1          1   zyxdim')
            + einops.rearrange(ray, '                   ray zyxdim -> 1 1 1          ray zyxdim')
            + einops.rearrange(offsets, '        neighbors zyxdim -> 1 1 neighbors 1   zyxdim')
        ).floor()  # y x neighbors ray zyx
        # directional distance in source volume coordinate system
        distance = pixel_rotated_y_x_zyx[:, :, None, None, :] - points_influencing_pixel
        # Inverse rotation projects this back to the original coordinate system, i.e
        # Distance in z is distance along the line, i.e. the slice profile weighted direction
        # Distance in x and y is the distance of a pixel to the ray and linear interpolation
        # is used to weight the distance
        distance_z, distance_y, distance_x = rotation(distance, inverse=True).unbind(-1)
        weight_yx = (1 - distance_y.abs()).clamp_min(0) * (1 - distance_x.abs()).clamp_min(0)
        weight_z = slice_function(distance_z)
        weight = (weight_yx * weight_z).reshape(y * x, -1)

        source = einops.rearrange(
            points_influencing_pixel,
            'y x neighbors raylength zyxdim -> (y x) (neighbors raylength) zyxdim',
        ).int()

        # mask of only potential source points inside the source volume
        mask = (
            (source[..., 0] < input_shape.z)
            & (source[..., 0] >= 0)
            & (source[..., 1] < input_shape.y)
            & (source[..., 1] >= 0)
            & (source[..., 2] < input_shape.x)
            & (source[..., 2] >= 0)
        )

        # We need this at the edge of the volume to approximate zero padding
        fraction_in_view = (mask * (weight > 0)).sum(-1) / (weight > 0).sum(-1)

        source_index = torch.tensor(
            np.ravel_multi_index(source[mask].unbind(-1), (input_shape.z, input_shape.y, input_shape.x))
        )
        target_index = torch.repeat_interleave(torch.arange(y * x), mask.sum(-1))

        with warnings.catch_warnings():
            # beta status in pytorch causes a warning to be printed
            warnings.filterwarnings('ignore', category=UserWarning, message='Sparse')

            matrix = torch.sparse_coo_tensor(
                indices=torch.stack((target_index, source_index)),
                values=weight.reshape(y * x, -1)[mask],
                size=(y * x, input_shape.z * input_shape.y * input_shape.x),
                dtype=torch.float32,
            ).coalesce()

            # To avoid giving more weight to points that are duplicated in our ray
            # logic and got summed in the coalesce operation, we normalize by the number
            # of duplicates. This is equivalent to the sum of the weights of the duplicates.
            # Count duplicates...

            ones = torch.ones_like(source_index).float()
            ones = torch.sparse_coo_tensor(
                indices=torch.stack((target_index, source_index)),
                values=ones,
                size=(y * x, input_shape.z * input_shape.y * input_shape.x),
                dtype=torch.float32,
            )
            # Coalesce sums the values of duplicate indices
            ones = ones.coalesce()

        # .. and normalize by the number of duplicates
        matrix.values()[:] /= ones.values()

        # Normalize for slice profile, so that the sum of the weights is 1
        # independent of the number of points that are considered.
        # Within the volume, the column sum should be 1,
        # at the edge of the volume, the column sum should be the fraction of the slice
        # that is in view to approximate zero padding
        norm = fraction_in_view / (matrix.sum(1).to_dense() + 1e-6)
        matrix *= norm[:, None]
        return matrix
