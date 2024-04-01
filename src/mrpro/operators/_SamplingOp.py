"""Class for Sampling Operator"""

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

import torch
from mrpro.operators import LinearOperator
from einops import rearrange
from mrpro.data import SpatialDimension

class AdjointGridSample(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  y, grid, xshape, interpolation_mode='bilinear', padding_mode='zeros', align_corners=True):
            """Adjoint of the linear operator x->gridsample(x,grid)

            Parameters
            ----------
            y
                tensor in the range of gridsample(x,grid). Should not include batch or channel dimension.
            grid
                grid in the shape (*y.shape, 2/3)
            xshape
                shape of the domain of gridsample(x,grid), i.e. the shape of x

            """
            ctx.interpolation_mode = interpolation_mode
            ctx.padding_mode = padding_mode
            ctx.align_corners = align_corners
            if y.requires_grad and grid.requires_grad:
                ctx.save_for_backward(grid, y)
            elif y.requires_grad:
                ctx.save_for_backward(grid)
            elif grid.requires_grad:
                ctx.save_for_backward(y)


            match interpolation_mode:
                case "bilinear":
                    mode_enum = 0
                case "nearest":
                    mode_enum = 1
                case "bicubic":
                    mode_enum = 2
                case _:
                    raise ValueError(f"Interpolation mode {interpolation_mode} not supported")
            match padding_mode:
                case "zeros":
                    padding_mode_enum = 0
                case "border":
                    padding_mode_enum = 1
                case "reflection":
                    padding_mode_enum = 2
                case _:
                    raise ValueError(f"Padding mode {padding_mode} not supported")

            match dim := grid.shape[-1]:
                case 3:
                    f = torch.ops.aten.grid_sampler_3d_backward
                case 2:
                    f = torch.ops.aten.grid_sampler_2d_backward
                case _:
                    raise ValueError(f"only 2d and 3d supported, not {dim}")

            if y.shape[0]!=grid.shape[0]:
                raise ValueError(f"y and grid must have same batch size, got {y.shape=}, {grid.shape=}")
            if xshape[1]!=y.shape[1]:
                raise ValueError(f"xshape and y must have same number of channels, got {xshape[1]} and {y.shape[1]}.")
            if len(xshape)-2 != dim:
                raise ValueError(f"len(xshape) and dim must either both bei 2 or 3, got {len(xshape)} and {dim}")

            shape_dummy = torch.empty(1,dtype=y.dtype, device=y.device).broadcast_to(xshape)
            x = f(
                y,
                shape_dummy,
                grid,
                interpolation_mode=mode_enum,
                padding_mode=padding_mode_enum,
                align_corners=align_corners,
                output_mask=[True, False],
            )[0]
            return x

    @staticmethod
    def backward(ctx, *grad_output):
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            y,grid, *_ = ctx.saved_tensors
        elif ctx.needs_input_grad[0]:
            grid, *_ = ctx.saved_tensors
            y = None
        elif ctx.needs_input_grad[1]:
            y, *_ = ctx.saved_tensors
            grid = None
        if ctx.needs_input_grad[0]:
            ygrad = torch.nn.functional.grid_sample(grad_output[0],grid,ctx.interpolation_mode, ctx.padding_mode, ctx.align_corners)
        else:
            ygrad = None
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("Gradients of the adjoint gridsample wrt the grid are not yet implemented.")
        else:
            gridgrad = None
        return ygrad, gridgrad, None, None, None ,None
        ...


class SamplingOp(LinearOperator):
    grid : torch.Tensor
    def __init__(self, grid:torch.Tensor, interpolation_mode="bilinear", padding_mode="zeros", align_corners=True, input_shape:None|SpatialDimension=None):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.padding_mode = padding_mode
        self.register_buffer("grid", grid)
        self.input_shape = input_shape
        self.align_corners = align_corners


    def forward(self,x:torch.Tensor)->tuple[torch.Tensor]:
        dim = self.grid.shape[-1]
        x_real = torch.view_as_real(x).moveaxis(-1,-dim-1) if x.is_complex() else x
        shape_grid_batch = self.grid.shape[:-dim-1]
        n_batchdim = len(shape_grid_batch)
        shape_x_batch = x_real.shape[:n_batchdim]
        shape_batch = torch.broadcast_shapes(shape_x_batch, shape_grid_batch)
        shape_channels = x_real.shape[n_batchdim:-dim]
        grid_flatbatch = self.grid.broadcast_to(*shape_batch,*self.grid.shape[n_batchdim:]).flatten(end_dim=n_batchdim-1)
        x_flatbatch = x_real.broadcast_to(*shape_batch, *x_real.shape[n_batchdim:]).flatten(end_dim=n_batchdim-1)
        x_flatbatch_flatchannel=x_flatbatch.flatten(start_dim=1,end_dim=-dim-1)
        sampled = torch.nn.functional.grid_sample(x_flatbatch_flatchannel,grid_flatbatch, mode=self.interpolation_mode, padding_mode=self.padding_mode, align_corners=self.align_corners)
        result = sampled.reshape(*shape_batch,*shape_channels, *sampled.shape[-dim:])
        if x.is_complex():
            result = torch.view_as_complex(result.moveaxis(-dim-1,-1).contiguous())
        return (result,)

    def adjoint(self,x:torch.Tensor)->tuple[torch.Tensor]:
        dim = self.grid.shape[-1]
        x_real = torch.view_as_real(x).moveaxis(-1,-dim-1) if x.is_complex() else x
        shape_grid_batch = self.grid.shape[:-dim-1]
        n_batchdim = len(shape_grid_batch)
        shape_x_batch = x_real.shape[:n_batchdim]
        shape_batch = torch.broadcast_shapes(shape_x_batch, shape_grid_batch)
        shape_channels = x_real.shape[n_batchdim:-dim]
        grid_flatbatch = self.grid.broadcast_to(*shape_batch,*self.grid.shape[n_batchdim:]).flatten(end_dim=n_batchdim-1)
        x_flatbatch = x_real.broadcast_to(*shape_batch, *x_real.shape[n_batchdim:]).flatten(end_dim=n_batchdim-1)
        x_flatbatch_flatchannel=x_flatbatch.flatten(start_dim=1,end_dim=-dim-1)
        if self.input_shape is None:
            domain = [int((i.max()-i.min()+1).item()) for i in self.grid.unbind(-1)]
        else:
            domain = self.input_shape.zyx[-dim:]
        shape = (*x_flatbatch_flatchannel.shape[:-dim],*domain)
        sampled = AdjointGridSample.apply(x_flatbatch_flatchannel,grid_flatbatch,shape, self.interpolation_mode, self.padding_mode,self.align_corners)
        result = sampled.reshape(*shape_batch,*shape_channels, *sampled.shape[-dim:])
        if x.is_complex():
            result = torch.view_as_complex(result.moveaxis(-dim-1,-1).contiguous())
        return (result,)



