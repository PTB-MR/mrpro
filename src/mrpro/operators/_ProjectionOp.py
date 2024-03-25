"""Class for 3D->2D Projection Operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import Literal

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

from mrpro.data import SpatialDimension


class _MatrixMultiplication(torch.autograd.Function):
    """Helper for matrix multiplication.

    For sparse matrices, it can be more efficient to have a
    separate representation of the adjoint matrix to be used
    in the backward pass.

    """

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, matrix: torch.Tensor, matrix_adjoint: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(matrix_adjoint)
        return matrix @ x

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        (matrix_adjoint,) = ctx.saved_tensors
        return matrix_adjoint @ grad_output, None, None


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

    def __call__(self, x):
        return torch.as_tensor(np.interp(x, self._xs.numpy(), self._weights.numpy(), 0, 0))


class SliceProjection(torch.nn.Module):
    def __init__(
        self,
        input_shape: Sequence[int, int, int] | SpatialDimension,
        slice_rotation: Rotation | Tensor,
        slice_shift: float | Tensor = 0.0,
        slice_profile: Callable[[Tensor], Tensor] | tuple[Callable[[Tensor], Tensor], ...] = SliceGaussian(2.0),
        optimize_for: Literal['forward', 'adjoint', 'both'] = 'both',
    ):
        """Create a module that represents the projection of a volume onto a plane.

        The projection will be done by sparse matrix multiplication.
        Either the slice_fwhm representing the slice thickness of a gaussian slice or
        the slice_x and slice_weight representing the slice profile must be given.

        Parameters
        ----------
        input_shape:
            Shape of the 3D volume to sample from
        slice_rotation
            Rotation that describes the orientation of the plane as a quaternion.
            If a tensor, it should be rotation quaternions.
        slice_shift
            Offset of the plane in the volume perpendicular plane from the center of the volume.
        slice_profile:
            A function that called with a distance x from the slice center should return the
            intensity along the slice thickness at x
        optimize_for: Literal["forward", "adjoint", "both"]
            Whether to optimize for forward or adjoint operation or both.
            Optimizing for both takes more memory but is faster for both operations

        """
        super().__init__()

        if slice_rotation is None:
            slice_rotation_quaternions = torch.tensor((0.0, 0.0, 0.0, 1))
        elif isinstance(slice_rotation, (tuple | list)) and all([isinstance(s, Rotation) for s in slice_rotation]):
            slice_rotation_quaternions = torch.stack(R.as_quat for R in slice_rotation)
        else:
            slice_rotation_quaternions = torch.as_tensor(slice_rotation)
            if not slice_rotation_quaternions.shape[-1] == 4:
                raise ValueError('Rotation quaternions must have 4 components')
        slice_rotation_quaternions = torch.atleast_2d(slice_rotation_quaternions)
        slice_shift = torch.atleast_1d(torch.as_tensor(slice_shift))

        batch_shapes = torch.broadcast_shapes(
            slice_rotation_quaternions.shape[:-1],
            slice_shift.shape,
        )

        if not isinstance(slice_profile, (tuple, list)):
            slice_profile = (slice_profile,) * np.prod(batch_shapes)
        elif len(slice_profile) == 1 and np.prod(batch_shapes) > 1:
            slice_profile = slice_profile * np.prod(batch_shapes)
        elif len(slice_profile) != np.prod(batch_shapes):
            raise ValueError('length of slice_profile does not match batch shapes')
        m = max(input_shape)

        ws = []
        for p in slice_profile:
            # figure out how far along the profile we have to consider values
            # clip up to 0.01 of intensity on both sides
            r = torch.arange(-m, m)
            pr = p(r)
            cs = torch.cumsum(pr, -1) / pr.sum()
            left = r[np.argmax(cs > 0.01)]
            right = r[np.argmax(cs > 0.99)]
            ws.append(int(max(left.abs(), right.abs()) + 1))
        slice_rotation_quaternions = slice_rotation_quaternions.expand(batch_shapes + (4,)).reshape(-1, 4)
        slice_shift = slice_shift.expand(batch_shapes).reshape(-1, 1)

        matrices = [
            SliceProjection.projection_matrix(
                input_shape,
                (m, m),
                offset=torch.tensor([0.0, 0.0, shift]),
                slice_function=f,
                rotation_quaternion=quat,
                w=int(w),
            )
            for quat, shift, f, w in zip(slice_rotation_quaternions, slice_shift, slice_profile, ws, strict=False)
        ]
        matrix = SliceProjection.join_matrices(matrices)

        # in csr format the matmul is faster, but saving one for forward and adjoint takes more memory
        if optimize_for == 'forward':
            self.matrix = matrix.to_sparse_csr()
            self.matrixT = self.matrix.H
        elif optimize_for == 'adjoint':
            self.matrixT = self.matrix.H.to_sparse_csr()
            self.matrix = self.matrixT.H
        elif optimize_for == 'both':
            self.matrix = matrix.to_sparse_csr()
            self.matrixT = self.matrix.H.to_sparse_csr()
        else:
            raise ValueError("optimize_for must be one of 'forward', 'adjoint', 'both'")

        self._range_shape = (*batch_shapes, m, m)
        self._domain_shape = input_shape

    def forward(self, x):
        x = _MatrixMultiplication().apply(x.ravel(), self.matrix, self.matrixT)
        return x.reshape(self._range_shape)

    def adjoint(self, x):
        x = _MatrixMultiplication().apply(x.ravel(), self.matrixT, self.matrix)
        return x.reshape(self._domain_shape)

    @staticmethod
    def join_matrices(matrices):
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
        matrix = torch.sparse_coo_tensor(
            indices=torch.stack([torch.cat(target), torch.cat(source)]),
            values=torch.cat(values),
            dtype=torch.float32,
            size=(len(matrices) * m.shape[0], m.shape[1]),
        )
        return matrix

    @staticmethod
    def projection_matrix(
        input_shape: tuple[int, int, int],
        output_shape: tuple[int, int],
        rotation_quaternion: Tensor,
        offset: Tensor,
        w: int,
        slice_function: Callable[[Tensor], Tensor],
        rotation_center=None,
    ):
        rotmat = torch.tensor(Rotation.from_quat(rotation_quaternion).as_matrix(), dtype=torch.float32)

        def _rotate(vector, inverse=False):
            if inverse:
                return (rotmat.T @ vector.reshape(-1, 3, 1)).reshape(vector.shape)
            else:
                return (rotmat @ vector.reshape(-1, 3, 1)).reshape(vector.shape)

        """Create a sparse matrix that represents the projection of a volume onto a plane

        Outside the volume values are approximately zero padded

        Parameters
        ----------
        input_shape:
            Shape of the volume to sample from
        output_shape:
            Shape of the resulting plane
        rotation_quaternion
            Rotation that describes the orientation of the plane as a quaternion
        offset: Tensor
            Offset of the plane in the volume in plane coordinates from the center of the volume
        w: int
            Factor that determines the number of pixels that are considered in the projection along the slice profile direction.
        slice_function: Callable
            Function that describes the slice profile
        rotation_center: Tensor
            Center of rotation, if None the center of the volume is used, i.e. for 4 pixels 0 1 2 3 it is between 1 and 2

        Returns
        -------
        torch.sparse_coo_matrix
            Sparse matrix that represents the projection of the volume onto the plane
        """
        X, Y, Z = input_shape
        x, y = output_shape  # a xy plane

        sx, sy = (
            (X - x) // 2,
            (Y - y) // 2,
        )  # coordinates of the 2d output pixels in the coordinate system of the input volume, so shape (x,y,3)

        pixel = torch.stack(
            [
                *torch.meshgrid(torch.arange(sx, sx + x), torch.arange(sy, sy + y)),  # x and y coordinates
                Z / 2 * torch.ones(x, y),  # z coordinates
            ],
            dim=-1,
        )
        if offset is not None:
            pixel = pixel + offset
        if rotation_center is None:
            # default rotation center is the center of the volume, i.e. for 4 pixels
            # 0 1 2 3 it is between 0 and 1
            rotation_center = torch.tensor([X / 2 - 0.5, Y / 2 - 0.5, Z / 2 - 0.5])
        pixel_rotated = _rotate(pixel - rotation_center) + rotation_center

        # We cast a ray from the pixel normal to the plane in both directions
        # points in the original volume further away then w will not be considered
        ray = _rotate(
            torch.stack(
                [
                    torch.zeros(2 * w + 1),  # X
                    torch.zeros(2 * w + 1),  # Y
                    torch.arange(-w, w + 1),  # Z
                ],
                dim=-1,
            )
        )
        # In all possible directions for each point aloing the line we consider the eight neighboring points
        # by adding all possible combinations of 0 and 1 to the point and flooring
        # (this is the same as adding -.5, .5 to the point and rounding)
        offsets = torch.tensor(list(itertools.product([0, 1], repeat=3)))
        # all points that influence a pixel
        # x,y,8-neighbours,(2*w+1)-raylength,3-dimensions XYZ)
        points_influencing_pixel = (
            einops.rearrange(pixel_rotated, '   x y XYZ -> x y 1          1   XYZ')
            + einops.rearrange(ray, '           ray XYZ -> 1 1 1          ray XYZ')
            + einops.rearrange(offsets, 'neighbours XYZ -> 1 1 neighbours 1   XYZ')
        ).floor()
        # directional distance in source volume coordinate system
        distance = pixel_rotated[:, :, None, None, :] - points_influencing_pixel
        # Inverse rotation projects this back to the original coordinate system, i.e
        # Distance in z is distance along the line, i.e. the slice profile weighted direction
        # Distance in x and y is the distance of a pixel to the ray and linear interpolation is used to weight the distance
        distance_x, distance_y, distance_z = _rotate(distance, inverse=True).unbind(-1)
        weight_xy = (1 - distance_x.abs()).clamp_min(0) * (1 - distance_y.abs()).clamp_min(0)
        weight_z = slice_function(distance_z)
        weight = (weight_xy * weight_z).reshape(x * y, -1)

        source = einops.rearrange(
            points_influencing_pixel,
            'x y neighbours raylength XYZdim -> (x y) (neighbours raylength) XYZdim',
        ).int()

        # mask of only potential source points inside the source volume
        mask = (
            (source[..., 0] < X)
            & (source[..., 0] >= 0)
            & (source[..., 1] < Y)
            & (source[..., 1] >= 0)
            & (source[..., 2] < Z)
            & (source[..., 2] >= 0)
        )

        # We need this at the edge of the volume to approximate zero padding
        fraction_in_view = (mask * (weight > 0)).sum(-1) / (weight > 0).sum(-1)

        source_index = torch.tensor(np.ravel_multi_index(source[mask].unbind(-1), (X, Y, Z)))
        target_index = torch.repeat_interleave(torch.arange(x * y), mask.sum(-1))

        # Count duplicates. Coalesce will sum the values of duplicate indices
        ones = torch.ones_like(source_index).float()
        ones = torch.sparse_coo_tensor(
            indices=torch.stack((target_index, source_index)),
            values=ones,
            size=(x * y, X * Y * Z),
            dtype=torch.float32,
        )
        ones = ones.coalesce()

        matrix = torch.sparse_coo_tensor(
            indices=torch.stack((target_index, source_index)),
            values=weight.reshape(x * y, -1)[mask],
            size=(x * y, X * Y * Z),
            dtype=torch.float32,
        ).coalesce()

        # To avoid giving to much weight to points that are duplicated in our logic and summed up by coalesce
        matrix.values()[:] /= ones.values()

        # Normalize
        norm = fraction_in_view / (matrix.sum(1).to_dense() + 1e-6)
        matrix *= norm[:, None]
        return matrix
