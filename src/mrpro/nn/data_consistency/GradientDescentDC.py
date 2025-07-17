import torch
from torch.nn import Module, Parameter

from mrpro.data.KData import KData
from mrpro.operators.LinearOperator import LinearOperator


class GradientDescentDC(Module):
    r"""Gradient descent data consistency.

    Performs gradient descent steps on
    :math:`\|Ax - k\|_2^2` where :math:`A` is the acquistion operator and :math:`k` is the data.

    Parameters
    ----------
    initial_stepsize
        Initial stepsize. The stepsize is a trainable parameter.
    n_steps
        Number of gradient descent steps.

    Returns
    -------
        The updated image.
    """

    def __init__(self, initial_stepsize: float, n_steps: int = 1) -> None:
        super().__init__()
        self.stepsize = Parameter(torch.tensor(initial_stepsize))
        self.n_steps = n_steps

    def forward(self, x: torch.Tensor, data: KData | torch.Tensor, acquistion_operator: LinearOperator) -> torch.Tensor:
        """Forward pass."""
        data_ = data.data if isinstance(data, KData) else data
        for _ in range(self.n_steps):
            residual = acquistion_operator(x)[0] - data_
            x = x - self.stepsize * acquistion_operator.adjoint(residual)[0]
        return x
