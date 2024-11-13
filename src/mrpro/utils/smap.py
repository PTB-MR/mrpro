"""Smap utility function."""

from collections.abc import Callable, Sequence

import torch


def smap(
    function: Callable[[torch.Tensor], torch.Tensor],
    tensor: torch.Tensor,
    passed_dimensions: Sequence[int] | int = (-1,),
) -> torch.Tensor:
    """Apply a function to a tensor serially along multiple dimensions.

    The function is applied serially without a batch dimensions.
    Compared to torch.vmap, it works with arbitrary functions, but is slower.

    Parameters
    ----------
    function
        Function to apply to the tensor.
        Should handle len(fun_dims) dimensions and not change the number of dimensions.
    tensor
        Tensor to apply the function to.
    passed_dimensions
        Dimensions NOT to be batched / dimensions that are passed to the function
        tuple of dimension indices (negative indices are supported) or an integer
        an integer n means the last n dimensions are passed to the function
    """
    if isinstance(passed_dimensions, int):
        # use the last fun_dims dimensions for the function
        moved = tensor
        first_fun_dim = -passed_dimensions
    else:
        # Move fun_dims to the end
        fun_dims_dst = tuple(range(-len(passed_dimensions), 0))
        moved = tensor.moveaxis(tuple(passed_dimensions), fun_dims_dst)
        first_fun_dim = fun_dims_dst[0]

    reshaped = moved.flatten(end_dim=first_fun_dim - 1)  # shape: (prod(batch_dims), fun_dim_1, ..., fun_dim_n)
    result_reshaped = torch.stack([function(x) for x in reshaped])
    result = result_reshaped.reshape(moved.shape[:first_fun_dim] + result_reshaped.shape[1:])

    if not isinstance(passed_dimensions, int):
        # Move fun_dims back to their original position if we moved them
        result = result.moveaxis(fun_dims_dst, tuple(passed_dimensions))
    return result
