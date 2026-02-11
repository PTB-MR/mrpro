"""Gradient descent data consistency."""

from typing import overload

import torch
from torch.nn import Module, Parameter

from mr2.data.KData import KData
from mr2.operators.FourierOp import FourierOp
from mr2.operators.LinearOperator import LinearOperator


class GradientDescentDC(Module):
    r"""Gradient descent data consistency.

    Performs gradient descent steps on
    :math:`\|Ax - k\|_2^2` where :math:`A` is the acquisition operator and :math:`k` is the data.

    Parameters
    ----------
    initial_stepsize
        Initial stepsize. The stepsize is a trainable parameter.
        Must be a positive scalar.
    n_steps
        Number of gradient descent steps.

    Returns
    -------
        The updated image.
    """

    def __init__(self, initial_stepsize: float | torch.Tensor, n_steps: int = 1) -> None:
        """Initialize the gradient descent data consistency.

        Parameters
        ----------
        initial_stepsize
            Initial stepsize. The stepsize is a trainable parameter.
            Must be a positive scalar.
        n_steps
            Number of gradient descent steps.
        """
        super().__init__()
        stepsize = torch.as_tensor(initial_stepsize)
        if stepsize.ndim != 0:
            raise ValueError('Stepsize must be a scalar')
        if stepsize.item() <= 0:
            raise ValueError('Stepsize must be positive')
        self.log_stepsize = Parameter(stepsize.log())
        self.n_steps = n_steps

    @overload
    def __call__(
        self, image: torch.Tensor, data: KData, acquisition_operator: LinearOperator | None = None
    ) -> torch.Tensor: ...

    @overload
    def __call__(
        self, image: torch.Tensor, data: torch.Tensor, acquisition_operator: LinearOperator
    ) -> torch.Tensor: ...

    def __call__(
        self, image: torch.Tensor, data: KData | torch.Tensor, acquisition_operator: LinearOperator | None = None
    ) -> torch.Tensor:
        """Apply the data consistency.

        Parameters
        ----------
        image
            Current image estimate.
        data
            k-space data.
        acquisition_operator
            Acquisition operator matching the k-space data. If None and data is provided as a `~mr2.data.KData`
            object, the Fourier operator is automatically created from the data.

        Returns
        -------
            Updated image estimate.
        """
        return super().__call__(image, data, acquisition_operator)

    def forward(
        self, image: torch.Tensor, data: KData | torch.Tensor, acquisition_operator: LinearOperator | None = None
    ) -> torch.Tensor:
        """Apply the data consistency."""
        if acquisition_operator is None:
            if isinstance(data, KData):
                acquisition_operator = FourierOp.from_kdata(data)
            else:
                raise ValueError('Either a KData or an acquisition operator is required')

        data_ = data.data if isinstance(data, KData) else data
        stepsize = self.log_stepsize.exp()
        x = image
        for _ in range(self.n_steps):
            residual = acquisition_operator(x)[0] - data_
            x = x - stepsize * acquisition_operator.adjoint(residual)[0]
        return x
