"""Smap utility function."""

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

from collections.abc import Callable

import torch


def smap(
    fun: Callable[[torch.Tensor], torch.Tensor],
    tensor: torch.Tensor,
    fun_dims: tuple[int, ...] | int = (-1,),
) -> torch.Tensor:
    """Apply a function to a tensor serially along multiple dimensions.

    The function is applied serially without a batch dimensions.
    Compared to torch.vmap, it works with arbitrary functions, but is slower.

    Parameters
    ----------
    fun
        Function to apply to the tensor.
        Should handle len(fun_dims) dimensions and not change the number of dimensions.
    tensor
        Tensor to apply the function to.
    fun_dims
        Dimensions NOT to be batched / dimensions that are passed to the function
        tuple of dimension indices (negative indices are supported) or an integer
        an integer n means the last n dimensions are passed to the function
    """
    if isinstance(fun_dims, int):
        # use the last fun_dims dimensions for the function
        moved = tensor
        first_fun_dim = -fun_dims
    else:
        # Move fun_dims to the end
        fun_dims_dst = tuple(range(-len(fun_dims), 0))
        moved = tensor.moveaxis(fun_dims, fun_dims_dst)
        first_fun_dim = fun_dims_dst[0]

    reshaped = moved.flatten(end_dim=first_fun_dim - 1)  # shape: (prod(batch_dims), fun_dim_1, ..., fun_dim_n)
    result_reshaped = torch.stack([fun(x) for x in reshaped])
    result = result_reshaped.reshape(moved.shape[:first_fun_dim] + result_reshaped.shape[1:])

    if not isinstance(fun_dims, int):
        # Move fun_dims back to their original position if we moved them
        result = result.moveaxis(fun_dims_dst, fun_dims)
    return result
