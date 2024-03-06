"""Class for Super Resolution Operator."""

# %%
# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import itertools
from collections.abc import Callable

import einops
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mrpro.operators import LinearOperator


class SuperResOp(LinearOperator):
    """Super resolution operator class."""

    def __init__(
        self,
        num_slices_per_stack: int,
        num_stacks: int,
        gap_slices_inHR: float,
        offsets_stack_HR: torch.Tensor,
        rot_per_stack: torch.Tensor,
        shape_HR: tuple,
        thickness_slice_inHR: float,
        w: int,
        slice_prof: Callable,
        rotation_center: torch.Tensor | None = None,
    ):
        """Super resolution operator class.

        Generate a sparse matrix that represents the projection of a volume
        onto a plane that is determined by R and offset.

        Parameters
        ----------
        input_shape: tuple
            shape of a volume in (x,y,z)
        R: scipy.spatial.transform.Rotation
            List of rotations of stacks that describes the orientation of the plane
        offset: torch.Tensor
            List of offsets of the slices of the plane in the volume
        w: int
            Factor that determines the number of pixels that are considered in the projection along the slice direction
        slice_function: Callable
            Function that describes the slice profile (weighting function, i.e. Gaussian)
        rotation_center: torch.Tensor
            Center of rotation, if None the center of the volume is used,
            i.e. for 4 pixels 0 1 2 3 it is between 1 and 2
        """

        self.num_stacks = num_stacks
        self.num_slices_per_stack = num_slices_per_stack
        self.shape_HR = shape_HR

        self.print(gap_slices_inHR=gap_slices_inHR, offsets_stack_HR=offsets_stack_HR, rot_per_stack=rot_per_stack)
        self.list_matrix = []

        # calc offsets
        range_slices = (
            self.num_slices_per_stack * thickness_slice_inHR + (self.num_slices_per_stack - 1) * gap_slices_inHR
        )
        offsets_slice = np.arange(0, range_slices + 1, step=thickness_slice_inHR + gap_slices_inHR, dtype=np.float32)
        offsets_slice -= np.mean(offsets_slice)

        R = []
        for idx_stack in range(self.num_stacks):
            R.append(Rotation.from_euler('xyz', rot_per_stack[idx_stack], degrees=True))

        for idx_stack in range(self.num_stacks):
            matrix_per_stack = []

            for idx_slice in range(self.num_slices_per_stack):
                offset_slicePos = torch.tensor([0, 0, offsets_slice[idx_slice] + offsets_stack_HR[idx_stack]])
                matrix_per_stack.append(
                    self._generate_matrix(
                        rot=R[idx_stack],
                        offset_slicePos=offset_slicePos,
                        length_normal=w,
                        slice_function=slice_prof,
                        rotation_center=rotation_center,
                    )
                )

            self.list_matrix.append(matrix_per_stack)

    def print(self, gap_slices_inHR: float, offsets_stack_HR: torch.Tensor, rot_per_stack: torch.Tensor) -> None:
        """Print settings of Super Res Op.

        Parameters
        ----------
        gap_slices_inHR
            gap between slices in unit of HR voxels
        offsets_stack_HR
            offsets between stack in unit of HR voxels
        rot_per_stack
            rotation between stacks in degrees
        """

        round_decimals = 2
        print('_______created super res op with the following parameters:_____')
        print('num_stacks = ' + str(self.num_stacks))
        print('num_slices_per_stack = ' + str(self.num_slices_per_stack))
        print('shape_HR = ' + str(self.shape_HR))
        print('gap_slices_inHR = ' + str(round(gap_slices_inHR, ndigits=round_decimals)))
        print('offsets_stack_HR = ' + str(torch.round(input=offsets_stack_HR, decimals=round_decimals)))
        print('rot_per_stack = ' + str(torch.round(input=rot_per_stack, decimals=round_decimals)))
        print('____________')

    def _rotate(self, vector: torch.Tensor, R: Rotation, rotation_center=None, inverse=False) -> torch.Tensor:
        """Rotate tensor by scipy Rotation around a rotation center."""

        ret = vector.reshape(-1, 3)
        if rotation_center is not None:
            ret = ret - rotation_center
        ret = torch.tensor(R.apply(ret.numpy(), inverse)).float()
        if rotation_center is not None:
            ret = ret + rotation_center
        return ret.reshape(vector.shape)

    def _generate_matrix(
        self, rot: Rotation, offset_slicePos: torch.Tensor, length_normal: int, slice_function, rotation_center=None
    ) -> torch.Tensor:
        """Sample sparse matrix representing projection of volume onto plane.

        Outside the volume it is zero padded.

        Returns
        -------
        torch.sparse_csr_matrix
            Sparse matrix that represents the projection operator of the volume onto the plane
        """

        X, Y, Z = self.shape_HR

        # select slice position
        pos_xy = list(torch.meshgrid(torch.arange(X), torch.arange(Y)))
        pos_z = [Z / 2 * torch.ones(X, Y)]

        pos = torch.stack(pos_xy + pos_z, dim=-1)
        pos = pos + offset_slicePos  # slice position is actually set according to offset

        if rotation_center is None:
            # default rotation center is the center of the volume, i.e. for 4 pixels
            # 0 1 2 3 it is between 0 and 1
            rotation_center = torch.tensor([X / 2 - 0.5, Y / 2 - 0.5, Z / 2 - 0.5])

        # rotate the slice
        pos_rot = self._rotate(pos, rot, rotation_center=rotation_center)

        # We cast a ray from the pixel perpendicular to the plane.
        # Pixels further away then w will not be considered
        sliceNormal = torch.stack(
            [
                torch.zeros(2 * length_normal),
                torch.zeros(2 * length_normal),
                torch.arange(-length_normal, length_normal),
            ],
            dim=-1,
        )
        sliceNormal = self._rotate(sliceNormal, rot)  # shape: [6,3]

        # binary counter representing all neighboring voxel for a specific position
        # all points that could influence to intensity of a pixel in the new slice
        # alternative would be looping over the neighboring voxels
        offsets_neighbors = torch.tensor(list(itertools.product([0, 1], repeat=3)))  # shape [8,3]

        # Distance between two pixels will always at least be 1
        # In all possible directions for each point along the line we consider the eight neighboring points
        # by adding all possible combinations of 0 and 1 to the point and flooring
        # dimensions of point and distance: [X, Y, 8 being the neighbors, 2*lenght_normal, 3 being the 3D position]
        points_influencing_pixel = (
            pos_rot[:, :, None, None, :]
            + sliceNormal[None, None, None, :, :]
            + offsets_neighbors[None, None, :, None, :]
        ).floor()
        # distance of each voxel within slice and within length_normal to the center of the rotated slice
        distance_rot = pos_rot[:, :, None, None, :] - points_influencing_pixel

        # Inverse rotation projects the distance back to the original coordinate system, i.e
        # Distance in z is distance along the line,
        # i.e. the slice profile weighted direction
        # Distance in x and y is the distance of a pixel to the ray
        # and linear interpolation is used to weight the distance
        distance_x, distance_y, distance_z = self._rotate(distance_rot, rot, inverse=True).unbind(-1)
        weight_xy = (1 - distance_x.abs()).clamp_min(0) * (1 - distance_y.abs()).clamp_min(0)
        weight_z = slice_function(distance_z)
        weight = (weight_xy * weight_z).reshape(
            X * Y, -1
        )  # weight is the same for each z pos, therefore only dim x and y

        # Mara: understand the next commented lines

        # Remove duplicates & points outside the volume
        # This is unfortunatly much slower than using torch.sparse.coo_tensor.coalesce later on
        # ids=torch.cat([torch.arange(len(source)).unsqueeze(1).expand(source.shape[:-1]).unsqueeze(-1),source],-1).reshape(-1,4)
        # _,inverse_idx,counts=torch.unique(ids,return_counts=True,return_inverse=True,dim=0)
        # counts = counts[inverse_idx].reshape(weight.shape).float()
        # weight = weight / counts.float()

        source = einops.rearrange(
            points_influencing_pixel, 'X Y neighbors raylength  XYZdim -> (X Y) (neighbors raylength ) XYZdim'
        ).int()

        # Remove points outside the volume
        # find points in source outside volume
        mask = (
            (source[..., 0] < X)
            & (source[..., 0] >= 0)
            & (source[..., 1] < Y)
            & (source[..., 1] >= 0)
            & (source[..., 2] < Z)
            & (source[..., 2] >= 0)
        )

        # Needed at the edge of the volume to approximate zero padding
        fraction_in_view = (mask * (weight > 0)).sum(-1) / (weight > 0).sum(-1)

        source_index = torch.tensor(np.ravel_multi_index(source[mask].unbind(-1), (X, Y, Z)))
        target_index = torch.repeat_interleave(torch.arange(X * Y), mask.sum(-1))
        target_source_stack = torch.stack((target_index, source_index))

        # Count duplicates.
        # ensuring that points counted twice don't get twice the weight
        # Coalesce will sum the values of duplicate indices
        ones = torch.ones_like(source_index).float()
        ones = torch.sparse_coo_tensor(
            indices=target_source_stack, values=ones, size=(X * Y, X * Y * Z), dtype=torch.float32
        )
        ones = ones.coalesce()

        matrix = torch.sparse_coo_tensor(
            indices=target_source_stack,
            values=weight.reshape(X * Y, -1)[mask],
            size=(X * Y, X * Y * Z),
            dtype=torch.float32,
        ).coalesce()
        matrix.values()[:] /= ones.values()

        # Normalize
        norm = fraction_in_view / matrix.sum(1).to_dense()

        matrix = matrix * norm[:, None]

        matrix = matrix.to_sparse_csr()
        # hier dann auch auch .H berechnen und als sparse speichern

        return matrix

    def forward(self, vol_HR: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the projection operatore to high resolution 3d volume.

        Parameters
        ----------
        img_hr
            image data tensor with dimensions (other coils z y x)

        Returns
        -------
            image data tensor with dimensions (other coils 1 y x)
            that represents 2d low resolution slice
        """
        other, coils, z, y, x = vol_HR.shape
        stacks: list[torch.Tensor] = []
        for idx_stack in range(self.num_stacks):
            list_slices = torch.zeros([self.num_slices_per_stack, other, coils, 1, y, x])
            vol_HR_flat = einops.rearrange(vol_HR, 'other coils z y x -> (other coils) (x y z)')
            for idx_slice in range(self.num_slices_per_stack):
                matrix = self.list_matrix[idx_stack][idx_slice]
                slice_LR = torch.zeros((vol_HR_flat.shape[0], matrix.shape[0]))
                for idx_coil in range(vol_HR_flat.shape[0]):
                    slice_LR[idx_coil] = matrix @ vol_HR_flat[idx_coil]
                list_slices[idx_slice] = einops.rearrange(
                    slice_LR, '(other coils) (x y) -> other coils 1 y x', coils=coils, y=y
                )

            stacks.append(list_slices)

        tensor_stacks = torch.stack(stacks)
        tensor_stacks = einops.rearrange(
            tensor_stacks,
            'num_stacks num_slices_per_stack other coils 1 y x -> (num_stacks num_slices_per_stack other coils 1 y x)',
            coils=coils,
            y=y,
        )
        return tuple(tensor_stacks)

    def adjoint(self, slices_LR: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of operator.

        Parameters
        ----------
        slices_LR
            LR slices in format 1, 1, other, coils, _, y, x

        Returns
        -------
            combined HR volume
        """
        other, coils, _, y, x = slices_LR[0][0].shape
        z, y, x = self.shape_HR
        vol_HR = torch.zeros((other, coils, z, y, x))
        vol_HR_flat = einops.rearrange(vol_HR, 'other coils z y x -> (other coils) (x y z)')

        flag_normalize = False
        if flag_normalize:
            weightSum_HR = torch.zeros(vol_HR_flat.shape[1])

            # normalize matrix for HR
            for idx_stack in range(self.num_stacks):
                for idx_slice in range(self.num_slices_per_stack):
                    # TODO: improve and calc sum without converting to dense
                    mat_dense = self.list_matrix[idx_stack][idx_slice].to_dense()
                    slice_LR = slices_LR[idx_stack][idx_slice]
                    slice_LR = einops.rearrange(slice_LR, 'other coils 1 y x -> (other coils) (x y)', coils=coils, y=y)
                    isZero = torch.where(slice_LR[0] < 0.00005)
                    mat_dense[isZero] = 0
                    weightSum_HR += mat_dense.sum(0)

        for idx_stack in range(self.num_stacks):
            for idx_slice in range(self.num_slices_per_stack):
                slice_LR = slices_LR[idx_stack][idx_slice]
                slice_LR = einops.rearrange(slice_LR, 'other coils 1 y x -> (other coils) (x y)', coils=coils, y=y)
                matrix = self.list_matrix[idx_stack][idx_slice]
                for idx_coil in range(vol_HR_flat.shape[0]):
                    vol_HR_flat[idx_coil] += matrix.H @ slice_LR[idx_coil]

        if flag_normalize:
            vol_HR_flat /= torch.where(weightSum_HR != 0.0, weightSum_HR.unsqueeze(0), 1.0)

        vol_HR = einops.rearrange(vol_HR_flat, '(other coils) (x y z) -> other coils z y x', coils=coils, y=y, x=x, z=z)
        return (vol_HR,)

    # def checkAdjointForward(self, slices_LR, vol_HR):
    #     test = 0
    #     # TODO: Check if y.H@A(x) = (AH(y).H@x holds for all x and y


class Slice_prof:
    """Slice Profile class."""

    def __init__(self, thickness_slice: float, sigma=None) -> None:
        """Slice Profile.

        Parameters
        ----------
        thickness_slice
            slice thickness in unit of HR slices
        sigma, optional
            describes how far slice profile should be considered (along z), by default None
        """
        self.thickness_slice = thickness_slice
        self.sigma = 1 if sigma is None else sigma

    def exp(self, distance_z: float):
        """Exponential slice profile.

        Parameters
        ----------
        distance_z
            distance to center of slice prof

        Returns
        -------
            slice profile weight at respective distance
        """
        # unit of distance_z: high resolution voxel
        return torch.exp(-(distance_z**2) / (2 * self.sigma**2))

    def rect(self, distance_z: float):
        """Rectangular slice profile.

        Parameters
        ----------
        distance_z
            distance to center of slice prof

        Returns
        -------
            slice profile weight at respective distance
        """
        if distance_z < self.thickness_slice / 2.0:
            return 1.0
        return 0.0
