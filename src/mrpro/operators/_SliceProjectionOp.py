"""Class for 3D->2D Projection Operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# input_shape.you may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANinput_shape.y KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import warnings
from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal
from typing import TypeAlias

import einops
import numpy as np
import torch
from numpy._typing import _NestedSequence as NestedSequence
from torch import Tensor

from mrpro.data import SpatialDimension
from mrpro.operators import LinearOperator
from mrpro.utils._Rotation import Rotation


class _MatrixMultiplication(torch.autograd.Function):
    """Helper for matrix multiplication.

    For sparse matrices, it can be more efficient to have a
    separate representation of the adjoint matrix to be used
    in the backward pass.

    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, matrix: torch.Tensor, matrix_adjoint: torch.Tensor
    ) -> torch.Tensor:
        ctx.save_for_backward(matrix_adjoint)
        ctx.x_is_complex = x.is_complex()  # type: ignore[attr-defined]
        if x.is_complex() == matrix.is_complex():
            return matrix @ x
        elif x.is_complex():
            return torch.complex(matrix @ x.real, matrix @ x.imag)
        else:
            return torch.complex(matrix.real @ x, matrix.imag @ x)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None]:
        (matrix_adjoint,) = ctx.saved_tensors  # type: ignore[attr-defined]
        if ctx.x_is_complex:  # type: ignore[attr-defined]
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


class SliceGaussian:
    """Gaussian Slice Profile."""

    def __init__(self, fwhm: float | Tensor):
        self.fwhm = torch.as_tensor(fwhm)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(x**2) / (0.36 * self.fwhm**2))


class SliceSmoothedRect:
    """Rectangular Slice Profile with smooth flanks.

    The smaller n, the smoother it is. For n<1, the FWHM might be wrong

    """

    def __init__(self, fwhm: float | Tensor, n: float | Tensor):
        self.n = n
        self.fwhm = fwhm

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 2 / self.fwhm
        return torch.erf(self.n * (1 - y)) + torch.erf(self.n * (1 + y))


class SliceInterpolate:
    """Slice Profile based on Interpolation of Measured Profile."""

    def __init__(self, xs: Tensor, weights: Tensor):
        self._xs = xs
        self._weights = weights

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(np.interp(x, self._xs.numpy(), self._weights.numpy(), 0, 0))


TensorFunction: TypeAlias = Callable[[Tensor], Tensor]
DefaultGaussianSliceProfile = SliceGaussian(2.0)


class SliceProjectionOp(LinearOperator):
    matrix: torch.Tensor | None
    matrix_adjoint: torch.Tensor | None

    def __init__(
        self,
        input_shape: SpatialDimension[int],
        slice_rotation: Rotation | None = None,
        slice_shift: float | Tensor = 0.0,
        slice_profile: TensorFunction | np.ndarray | NestedSequence[TensorFunction] | float = 1.0,
        optimize_for: Literal['forward', 'adjoint', 'both'] = 'both',
    ):
        """Create a module that represents the 'projection' of a volume onto a plane.

        This operation samples from a 3D Volume a slice with a given rotation and shift,
        according to the slice_profile. It can, for example, be used to describe the slice
        selection of a 2D MRI sequence from the 3D Volume.-

        The projection will be done by sparse matrix multiplication.


        Parameters
        ----------
        input_shape:
            Shape of the 3D volume to sample from
        slice_rotation
            Rotation that describes the orientation of the plane. If None,
            an identity rotation is used.
        slice_shift
            Offset of the plane in the volume perpendicular plane from the center of the volume.
        slice_profile:
            A function that called with a distance x from the slice center should return the
            intensity along the slice thickness at x. This can also be a nested Sequence or an
            numpy array of functions.
            If it is a single float, it will be interpreted as a Gaussian FWHM.
        optimize_for: Literal["forward", "adjoint", "both"]
            Whether to optimize for forward or adjoint operation or both.
            Optimizing for both takes more memory but is faster for both operations.

        """
        super().__init__()
        if isinstance(slice_profile, float):
            slice_profile = SliceGaussian(slice_profile)
        slice_profile_array = np.array(slice_profile)

        if slice_rotation is None:
            slice_rotation = Rotation.identity()

        m = max(input_shape.z, input_shape.y, input_shape.x)

        def _find_width(slice_profile: TensorFunction) -> int:
            # figure out how far along the profile we have to consider values
            # clip up to 0.01 of intensity on both sides
            r = torch.arange(-m, m)
            profile = slice_profile(r)
            cdf = torch.cumsum(profile, -1) / profile.sum()
            left = r[np.argmax(cdf > 0.01)]
            right = r[np.argmax(cdf > 0.99)]
            return int(max(left.abs().item(), right.abs().item()) + 1)

        widths = np.vectorize(_find_width)(slice_profile_array)
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
                SpatialDimension(1, m, m),
                offset=torch.tensor([0.0, 0.0, shift]),
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

        self._range_shape: tuple[int] = (*batch_shapes, m, m)
        self._domain_shape = input_shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        match (self.matrix, self.matrix_adjoint):
            case (None, None):
                raise RuntimeError('Either matrix or matrix adjoint must be set')
            case (matrix, None) if matrix is not None:
                matrix_adjoint = matrix.H
            case (None, matrix_adjoint) if matrix_adjoint is not None:
                matrix = matrix_adjoint.H
            case (matrix, matrix_adjoint):
                ...
        x = _MatrixMultiplication.apply(x.ravel(), matrix, matrix_adjoint)
        return (x.reshape(self._range_shape),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        match (self.matrix, self.matrix_adjoint):
            case (None, None):
                raise RuntimeError('Either matrix or matrix adjoint must be set')
            case (matrix, None) if matrix is not None:
                matrix_adjoint = matrix.H
            case (None, matrix_adjoint) if matrix_adjoint is not None:
                matrix = matrix_adjoint.H
            case (matrix, matrix_adjoint):
                ...
        x = _MatrixMultiplication.apply(x.ravel(), matrix_adjoint, matrix)
        return (x.reshape(self._domain_shape.z, self._domain_shape.y, self._domain_shape.x),)

    @staticmethod
    def join_matrices(matrices: Sequence[torch.Tensor]) -> torch.Tensor:
        values = []
        target = []
        source = []
        for i, m in enumerate(matrices):
            if not m.shape == matrices[0].shape:
                raise ValueError('all matrices should have the same shape')
            c = m.coalesce()
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
        rotation_center: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Create a sparse matrix that represents the projection of a volume onto a plane.

        Outside the volume values are approximately zero padded

        Parameters
        ----------
        input_shape:
            Shape of the volume to sample from
        output_shape:
            Shape of the resulting plane, 2D.
        rotation
            Rotation that describes the orientation of the plane
        offset: Tensor
            Offset of the plane in the volume in plane coordinates from the center of the volume
        w: int
            Factor that determines the number of pixels that are considered in the projection along
            the slice profile direction.
        slice_function: Callable
            Function that describes the slice profile
        rotation_center: Tensor
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
                input_shape.z / 2 * torch.ones(y, x),  # z coordinates
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
            rotation_center = torch.tensor([input_shape.x / 2 - 0.5, input_shape.y / 2 - 0.5, input_shape.z / 2 - 0.5])
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
        # In all possible directions for each point aloing the line we consider the eight neighboring points
        # by adding all possible combinations of 0 and 1 to the point and flooring
        # (this is the same as adding -.5, .5 to the point and rounding)
        offsets = torch.tensor(list(itertools.product([0, 1], repeat=3)))
        # all points that influence a pixel
        # x,y,8-neighbours,(2*w+1)-raylength,3-dimensions input_shape.xinput_shape.yinput_shape.z)
        points_influencing_pixel = (
            einops.rearrange(pixel_rotated_y_x_zyx, '   y x zyxdim -> y x 1          1   zyxdim')
            + einops.rearrange(ray, '                   ray zyxdim -> 1 1 1          ray zyxdim')
            + einops.rearrange(offsets, '        neighbours zyxdim -> 1 1 neighbours 1   zyxdim')
        ).floor()  # y x neighbours ray zyx
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
            'y x neighbours raylength zyxdim -> (y x) (neighbours raylength) zyxdim',
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

            # Count duplicates. Coalesce will sum the values of duplicate indices
            ones = torch.ones_like(source_index).float()
            ones = torch.sparse_coo_tensor(
                indices=torch.stack((target_index, source_index)),
                values=ones,
                size=(y * x, input_shape.z * input_shape.y * input_shape.x),
                dtype=torch.float32,
            )
            ones = ones.coalesce()

            matrix = torch.sparse_coo_tensor(
                indices=torch.stack((target_index, source_index)),
                values=weight.reshape(y * x, -1)[mask],
                size=(y * x, input_shape.z * input_shape.y * input_shape.x),
                dtype=torch.float32,
            ).coalesce()

        # To avoid giving to much weight to points that are duplicated in our logic and summed up by coalesce
        matrix.values()[:] /= ones.values()

        # Normalize
        norm = fraction_in_view / (matrix.sum(1).to_dense() + 1e-6)
        matrix *= norm[:, None]
        return matrix
