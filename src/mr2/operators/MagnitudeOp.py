"""Operator returning the magnitude of the input."""

import torch

from mr2.operators.EndomorphOperator import EndomorphOperator, endomorph


class MagnitudeOp(EndomorphOperator):
    """Magnitude of input tensors."""

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Calculate the element-wise magnitude (absolute value) of input tensors.

        Parameters
        ----------
        *x
            One or more input tensors.

        Returns
        -------
            A tuple of tensors, where each tensor contains the element-wise
            magnitude of the corresponding input tensor.
        """
        return super().__call__(*x)

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of MagnitudeOp.

        .. note::
            Prefer calling the instance of the MagnitudeOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return tuple([torch.abs(xi) for xi in x])
