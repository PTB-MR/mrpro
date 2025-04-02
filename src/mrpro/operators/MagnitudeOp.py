"""Operator returning the magnitude of the input."""

import torch

from mrpro.operators.EndomorphOperator import EndomorphOperator, endomorph


class MagnitudeOp(EndomorphOperator):
    """Magnitude of input tensors."""

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply MagnitudeOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        return tuple([torch.abs(xi) for xi in x])
