"""Operator returning the phase of the input."""

import torch

from mr2.operators.EndomorphOperator import EndomorphOperator, endomorph


class PhaseOp(EndomorphOperator):
    """Phase of input tensors."""

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Calculate the element-wise phase (angle) of input tensors.

        Parameters
        ----------
        *x
            One or more input tensors. Can be complex or real.

        Returns
        -------
            A tuple of tensors, where each tensor contains the element-wise
            phase of the corresponding input tensor. The phase is in radians.
        """
        return super().__call__(*x)

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of PhaseOp.

        .. note::
            Prefer calling the instance of the PhaseOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return tuple([torch.angle(xi) for xi in x])
