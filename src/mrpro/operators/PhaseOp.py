"""Operator returning the phase of the input."""

import torch

from mrpro.operators.EndomorphOperator import EndomorphOperator, endomorph


class PhaseOp(EndomorphOperator):
    """Phase of input tensors."""

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Phase of tensors.

        Parameters
        ----------
        x
            input tensors

        Returns
        -------
            tensors with phase of input tensors
        """
        return tuple([torch.angle(xi) for xi in x])
