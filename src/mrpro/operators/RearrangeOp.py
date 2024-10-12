"""Rearrange Operator."""

import re

import torch
from einops import rearrange

from mrpro.operators.LinearOperator import LinearOperator


class RearrangeOp(LinearOperator):
    """A Linear Operator that implements rearranging of axes.

    Wraps the einops.rearrange function to rearrange the axes of a tensor.
    """

    def __init__(self, pattern: str, additional_info: dict[str, int] | None = None) -> None:
        """Initialize Einsum Operator.

        Parameters
        ----------
        pattern
            Pattern describing the forward of the operator.
            Also see einops.rearrange for more information.
            Example: "... h w -> ... (w h)"
        additional_info
            Additional information passed to the rearrange function,
            describing the size of certain dimensions.
            Might be required for the adjoint rule.
            Example: {'h': 2, 'w': 2}
        """
        super().__init__()
        if (match := re.match('(.+)->(.+)', pattern)) is None:
            raise ValueError(f'pattern should match (.+)->(.+) but got {pattern}.')
        input_pattern, output_pattern = match.groups()
        # swapping the input and output gets the adjoint rule
        self._adjoint_pattern = f'{output_pattern}->{input_pattern}'
        self._forward_pattern = pattern
        self.additional_info = {} if additional_info is None else additional_info

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Rearrange input.

        The rule used to perform the rearranging is set at initialization.

        Parameters
        ----------
        x
            input tensor to be rearranged

        Returns
        -------
            rearranged tensor
        """
        y = rearrange(x, self._forward_pattern, **self.additional_info)
        return (y,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Rearrange input with the adjoint rule.

        Parameters
        ----------
        y
            tensor to be rearranged

        Returns
        -------
            rearranged tensor
        """
        x = rearrange(y, self._adjoint_pattern, **self.additional_info)
        return (x,)
