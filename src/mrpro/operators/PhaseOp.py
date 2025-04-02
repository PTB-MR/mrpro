"""Operator returning the phase of the input."""

import torch

from mrpro.operators.EndomorphOperator import EndomorphOperator, endomorph


class PhaseOp(EndomorphOperator):
    """Phase of input tensors."""

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply PhaseOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        return tuple([torch.angle(xi) for xi in x])
