"""Identity Operator with arbitrary number of inputs."""

import torch
from typing_extensions import Self

from mrpro.operators.EndomorphOperator import EndomorphOperator, endomorph


class MultiIdentityOp(EndomorphOperator):
    r"""The Identity Operator.

    An endomorph Operator that returns multiple inputs unchanged.
    """

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply MultiIdentityOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        return x

    @endomorph
    def adjoint(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Adjoint Identity.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            the input tensor
        """
        return x

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint Identity."""
        return self
