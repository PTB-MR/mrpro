"""Operator returning the magnitude of the input."""

import torch

from mrpro.operators.EndomorphOperator import EndomorphOperator, endomorph


class MagnitudeOp(EndomorphOperator):
    """Magnitude of input tensors."""

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Magnitude of tensors.

        Parameters
        ----------
        x
            input tensors

        Returns
        -------
            tensors with magnitude (absolute values) of input tensors
        """
        return tuple([torch.abs(xi) for xi in x])
