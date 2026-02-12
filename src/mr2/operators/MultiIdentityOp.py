"""Identity Operator with arbitrary number of inputs."""

import torch
from typing_extensions import Self

from mr2.operators.EndomorphOperator import EndomorphOperator, endomorph


class MultiIdentityOp(EndomorphOperator):
    r"""The Identity Operator.

    An endomorph Operator that returns multiple inputs unchanged.
    """

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the multi-input identity operation.

        This operator returns all input tensors unchanged.

        Parameters
        ----------
        *x
            One or more input tensors.

        Returns
        -------
            The input tensors, unchanged.
        """
        return super().__call__(*x)

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of MultiIdentityOp.

        .. note::
            Prefer calling the instance of the MultiIdentityOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return x

    @endomorph
    def adjoint(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the adjoint of the multi-input identity operation.

        Since the multi-input identity operator is self-adjoint, this method
        returns all input tensors unchanged.

        Parameters
        ----------
        *x
            One or more input tensors.

        Returns
        -------
            The input tensors, unchanged.
        """
        return x

    @property
    def H(self) -> Self:  # noqa: N802
        """Adjoint Identity."""
        return self
