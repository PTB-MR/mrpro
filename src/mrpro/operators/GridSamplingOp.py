"""Class for Grid Sampling Operator."""

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

import warnings
from collections.abc import Callable, Sequence
from typing import Literal

import torch

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators.LinearOperator import LinearOperator


class AdjointGridSample(torch.autograd.Function):
    """Autograd Function for Adjoint Grid Sample.

    Ensures that the Adjoint Operation is differentiable.

    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        y: torch.Tensor,
        grid: torch.Tensor,
        xshape: Sequence[int],
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = True,
    ) -> torch.Tensor:
        """Adjoint of the linear operator x->gridsample(x,grid).

        Parameters
        ----------
        ctx
            Context
        y
            tensor in the range of gridsample(x,grid). Should not include batch or channel dimension.
        grid
            grid in the shape (*y.shape, 2/3)
        xshape
            shape of the domain of gridsample(x,grid), i.e. the shape of x
        interpolation_mode
            the kind of interpolation used
        padding_mode
            how to pad the input
        align_corners
             if True, the corner pixels of the input and output tensors are aligned,
             and thus preserving the values at those pixels

        """
        # grid_sampler_and_backward uses integer values instead of strings for the modes
        match interpolation_mode:
            case 'bilinear':
                mode_enum = 0
            case 'nearest':
                mode_enum = 1
            case 'bicubic':
                mode_enum = 2
            case _:
                raise ValueError(f'Interpolation mode {interpolation_mode} not supported')

        match padding_mode:
            case 'zeros':
                padding_mode_enum = 0
            case 'border':
                padding_mode_enum = 1
            case 'reflection':
                padding_mode_enum = 2
            case _:
                raise ValueError(f'Padding mode {padding_mode} not supported')

        match dim := grid.shape[-1]:
            case 3:
                backward_2d_or_3d = torch.ops.aten.grid_sampler_3d_backward
            case 2:
                backward_2d_or_3d = torch.ops.aten.grid_sampler_2d_backward
            case _:
                raise ValueError(f'only 2d and 3d supported, not {dim}')

        if y.shape[0] != grid.shape[0]:
            raise ValueError(f'y and grid must have same batch size, got {y.shape=}, {grid.shape=}')
        if xshape[1] != y.shape[1]:
            raise ValueError(f'xshape and y must have same number of channels, got {xshape[1]} and {y.shape[1]}.')
        if len(xshape) - 2 != dim:
            raise ValueError(f'len(xshape) and dim must either both bei 2 or 3, got {len(xshape)} and {dim}')

        # These are required in the backward.
        ctx.xshape = xshape  # type: ignore[attr-defined]
        ctx.interpolation_mode = mode_enum  # type: ignore[attr-defined]
        ctx.padding_mode = padding_mode_enum  # type: ignore[attr-defined]
        ctx.align_corners = align_corners  # type: ignore[attr-defined]
        ctx.backward_2d_or_3d = backward_2d_or_3d  # type: ignore[attr-defined]
        if grid.requires_grad:
            # only if we need to calculate the gradient for grid we need y
            ctx.save_for_backward(grid, y)
        else:
            ctx.save_for_backward(grid)
        dummy = torch.empty(1, dtype=y.dtype, device=y.device).broadcast_to(xshape)
        x = backward_2d_or_3d(
            y,
            dummy,  # only the shape, device and dtype are relevant
            grid,
            interpolation_mode=mode_enum,
            padding_mode=padding_mode_enum,
            align_corners=align_corners,
            output_mask=[True, False],
        )[0]
        return x

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, *grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None, None, None]:
        """Backward of the Adjoint Gridsample Operator."""
        need_y_grad, need_grid_grad, *_ = ctx.needs_input_grad  # type: ignore[attr-defined]
        grid = ctx.saved_tensors[0]  # type: ignore[attr-defined]

        if need_y_grad:
            # torch.grid_sampler has the same signature as the backward
            # (and is used inside F.grid_sample)
            grad_y = torch.grid_sampler(
                grad_output[0],
                grid,
                ctx.interpolation_mode,  # type: ignore[attr-defined]
                ctx.padding_mode,  # type: ignore[attr-defined]
                ctx.align_corners,  # type: ignore[attr-defined]
            )
        else:
            grad_y = None

        if need_grid_grad:
            y = ctx.saved_tensors[1]  # type: ignore[attr-defined]
            grad_grid = ctx.backward_2d_or_3d(  # type: ignore[attr-defined]
                y,
                grad_output[0],
                grid,
                interpolation_mode=ctx.interpolation_mode,  # type: ignore[attr-defined]
                padding_mode=ctx.padding_mode,  # type: ignore[attr-defined]
                align_corners=ctx.align_corners,  # type: ignore[attr-defined]
                output_mask=[False, True],
            )[1]
        else:
            grad_grid = None

        return grad_y, grad_grid, None, None, None, None


class GridSamplingOp(LinearOperator):
    """Grid Sampling Operator.

    Given an "input" tensor and a "grid", computes the output by taking the input values at the locations
    determined by grid with interpolation. Thus, the output size will be determined by the grid size.
    For the adjoint to be defined, the grid and the shape of the "input" has to be known.
    """

    grid: torch.Tensor

    def __init__(
        self,
        grid: torch.Tensor,
        input_shape: SpatialDimension,
        interpolation_mode: Literal['bilinear', 'nearest', 'bicubic'] = 'bilinear',
        padding_mode: Literal['zeros', 'border', 'reflection'] = 'zeros',
        align_corners: bool = False,
    ):
        """Initialize Sampling Operator.

        Parameters
        ----------
        grid
            sampling grid. Shape *batchdim, z,y,x,3 / *batchdim, y,x,2.
            Values should be in [-1, 1.]
        input_shape
            Used in the adjoint. The z,y,x shape of the domain of the operator.
            If grid has 2 as the last dimension, only y and x will be used.
        interpolation_mode
            mode used for interpolation. bilinear is trilinear in 3D, bicubic is only supported in 2D.
        padding_mode
            how the input of the forward is padded.
        align_corners
            if True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at those pixels
        """
        super().__init__()

        match grid.shape[-1]:
            case 2:  # 2D
                if grid.ndim < 4:
                    raise ValueError(
                        'For a 2D gridding (determined by last dimension of grid), grid should have at least'
                        f' 4 dimensions: batch y x 2. Got shape {grid.shape}.'
                    )
            case 3:  # 3D
                if grid.ndim < 5:
                    raise ValueError(
                        'For a 3D gridding (determined by last dimension of grid), grid should have at least'
                        f' 5 dimensions: batch z y x 3. Got shape {grid.shape}.'
                    )
                if interpolation_mode == 'bicubic':
                    raise NotImplementedError('Bicubic only implemented for 2D')
            case _:
                raise ValueError('Grid should have 2 or 3 as last dimension for 2D or 3D sampling')
        if not grid.is_floating_point():
            raise ValueError(f'Grid should be a real floating dtype, got {grid.dtype}')
        if grid.max() > 1.0 or grid.min() < -1.0:
            warnings.warn('Grid has values outside range [-1,1].', stacklevel=1)

        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.register_buffer('grid', grid)
        self.input_shape = input_shape
        self.align_corners = align_corners

    def __reshape_wrapper(
        self, x: torch.Tensor, inner: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> tuple[torch.Tensor]:
        """Do all the reshaping pre- and post- sampling."""
        # First, we need to do a lot of reshaping ..
        dim = self.grid.shape[-1]
        if x.ndim < dim + 2:
            raise ValueError(
                f'For a {dim}D sampling operation, x should have at least have {dim+2} dimensions:'
                f' batch channel {"z y x" if dim==3 else "y x"}.'
            )

        #   The gridsample operator only works for real data, thus we handle complex inputs as an additional channel
        x_real = torch.view_as_real(x).moveaxis(-1, -dim - 1) if x.is_complex() else x
        shape_grid_batch = self.grid.shape[: -dim - 1]  # the batch dimensions of grid
        n_batchdim = len(shape_grid_batch)
        shape_x_batch = x_real.shape[:n_batchdim]  # the batch dimensions of the input
        try:
            shape_batch = torch.broadcast_shapes(shape_x_batch, shape_grid_batch)
        except RuntimeError:
            raise ValueError(
                'Batch dimensions in x and grid are not broadcastable.'
                f' Got batch dimensions x: {shape_x_batch} and grid: {shape_grid_batch},'
                f' (shapes are x: {x.shape}, grid: {self.grid.shape}).'
            ) from None

        shape_channels = x_real.shape[n_batchdim:-dim]
        #   reshape to 3D: (*batch_dim) z y x 3 or 2D: (*batch_dim) y x 2
        grid_flatbatch = self.grid.broadcast_to(*shape_batch, *self.grid.shape[n_batchdim:]).flatten(
            end_dim=n_batchdim - 1
        )
        x_flatbatch = x_real.broadcast_to(*shape_batch, *x_real.shape[n_batchdim:]).flatten(end_dim=n_batchdim - 1)
        #   reshape to 3D: (*batch_dim) (*channel_dim) z y x or 2D: (*batch_dim) (*channel_dim) y x
        x_flatbatch_flatchannel = x_flatbatch.flatten(start_dim=1, end_dim=-dim - 1)

        # .. now we can perform the actual sampling implementation..
        sampled = inner(x_flatbatch_flatchannel, grid_flatbatch)

        # .. and reshape back.
        result = sampled.reshape(*shape_batch, *shape_channels, *sampled.shape[-dim:])
        if x.is_complex():
            result = torch.view_as_complex(result.moveaxis(-dim - 1, -1).contiguous())
        return (result,)

    def _forward_implementation(
        self, x_flatbatch_flatchannel: torch.Tensor, grid_flatbatch: torch.Tensor
    ) -> torch.Tensor:
        """Apply the actual forward after reshaping."""
        sampled = torch.nn.functional.grid_sample(
            x_flatbatch_flatchannel,
            grid_flatbatch,
            mode=self.interpolation_mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )

        return sampled

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the GridSampleOperator.

        Samples at the location determine by the grid.
        """
        if (
            (x.shape[-1] != self.input_shape.x)
            or (x.shape[-2] != self.input_shape.y)
            or (x.shape[-3] != self.input_shape.z and self.grid.shape[-1] == 3)
        ):
            warnings.warn(
                'Mismatch between x.shape and input shape. Adjoint on the result will return the wrong shape.',
                stacklevel=1,
            )
        return self.__reshape_wrapper(x, self._forward_implementation)

    def _adjoint_implementation(
        self, x_flatbatch_flatchannel: torch.Tensor, grid_flatbatch: torch.Tensor
    ) -> torch.Tensor:
        """Apply the actual adjoint after reshaping."""
        dim = self.grid.shape[-1]
        shape = (*x_flatbatch_flatchannel.shape[:-dim], *self.input_shape.zyx[-dim:])
        sampled = AdjointGridSample.apply(
            x_flatbatch_flatchannel,
            grid_flatbatch,
            shape,
            self.interpolation_mode,
            self.padding_mode,
            self.align_corners,
        )
        return sampled

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the adjoint of the GridSampleOperator."""
        return self.__reshape_wrapper(x, self._adjoint_implementation)
