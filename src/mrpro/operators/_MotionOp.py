"""Motion transformation operator."""

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

from __future__ import annotations

import torch
import torch.nn.functional as torch_func
from einops import rearrange

from mrpro.data import DisplacementField
from mrpro.operators import LinearOperator


class MotionOp(LinearOperator):
    def __init__(
        self,
        displacement_field: DisplacementField,
    ) -> None:
        """Motion Operator class.

        This operator takes a displacement field with the displacement in voxel and calculates a deformation field/flow
        field suitable for torch.grid_sample. Torch.grid_sample requires a deformation field which specifies the
        sampling voxel locations normalized by the input spatial dimensions.

        Parameters
        ----------
        displacement_field
            displacement field describing spatial motion transformations
        """
        super().__init__()

        # If other > 1 then the displacement fields are not broadcasted along this dimension and this dimension needs
        # to be added to the motion_state dimension
        if displacement_field.fz.shape[1] > 1:
            self.broadcast_to_other = False
        else:
            self.broadcast_to_other = True

        # Combine other and motion_states. -1 is needed because here we define the resampling grid, so if we want an
        # object to move by a dx, we have to calculate a grid which starts -dx before the current grid.
        self.grid = -rearrange(displacement_field.as_tensor(stack_dim=-1), 'ms other z y x dim->(ms other) z y x dim')

        # Normalize the displacement by the input spatial dimensions. Factor two is needed because grid is defined
        # between [-1, 1]
        self.grid *= 2 / rearrange(torch.tensor(displacement_field.fz.shape[-3:]), 'dim->1 1 1 1 dim')

        # A vector in the displacement field is defined as [z, y, x].
        # torch.resample expects the grid vector as [x, y, z].
        self.grid = torch.flip(self.grid, dims=[-1])

        # Create a grid between [-1, 1] with the dimensions (motion_state, z, y, x)
        unity_matrix = torch.cat((torch.eye(3), torch.zeros(3, 1)), dim=1)
        grid_size = (1, 1) + self.grid.shape[1:-1]  # same unity grid for all motion states and batches
        unity_grid = torch_func.affine_grid(unity_matrix[None, ...], size=grid_size, align_corners=False)

        # Add grid to displacement
        self.grid += unity_grid

        # Calculate inverse grid
        self.inverse_grid = torch.empty(1)

    def combine_ms_other(self, x: torch.Tensor) -> torch.Tensor:
        """Combine motion states and other.

        Parameters
        ----------
        x
            input data (motion_states, other, z, y, x)

        Returns
        -------
            data with motion_states and other combined if broadcast_to_other is False
        """
        if not self.broadcast_to_other:
            x = rearrange(x, 'ms other z y x->(ms other) z y x')
        return x

    def split_ms_other(self, x: torch.Tensor) -> torch.Tensor:
        """Split motion states and other.

        Parameters
        ----------
        x
            input data ((motion_states, other), z, y, x)

        Returns
        -------
            data with motion_states and other split if broadcast_to_other is False
        """
        if not self.broadcast_to_other:
            x = rearrange(x, '(ms other) z y x->ms other z y x')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 5:
            raise ValueError('The input tensor should have 5 dimensions (motion_states, other, z, y, x).')
        return self.split_ms_other(
            torch_func.grid_sample(self.combine_ms_other(x), self.grid, padding_mode='border', align_corners=False)
        )

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) != 5:
            raise ValueError('The input tensor should have 5 dimensions (motion_states, other, z, y, x).')
        return self.split_ms_other(
            torch_func.grid_sample(
                self.combine_ms_other(x), self.inverse_grid, padding_mode='border', align_corners=False
            )
        )
