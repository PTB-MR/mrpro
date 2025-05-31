"""Inversion recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils.reshape import unsqueeze_right


class InversionRecovery(SignalModel[torch.Tensor, torch.Tensor]):
    """Inversion recovery signal model."""

    def __init__(self, ti: float | torch.Tensor | Sequence[float]):
        """Initialize inversion recovery signal model for T1 mapping.

        Parameters
        ----------
        ti
            inversion times
            with shape (time, ...)
        """
        super().__init__()
        ti = torch.as_tensor(ti)
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def __call__(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the inversion recovery signal model.

        Calculates the signal based on the formula:
        S(TI) = M0 * (1 - 2 * exp(-TI / T1))
        where TI are the inversion times.

        Parameters
        ----------
        m0
            Equilibrium signal or proton density.
            Expected shape `(*other, coils, z, y, x)`.
        t1
            Longitudinal relaxation time T1.
            Expected shape `(*other, coils, z, y, x)`.

        Returns
        -------
        tuple[torch.Tensor,]
            Signal calculated for each inversion time.
            Shape `(time, *other, coils, z, y, x)`, where `time` corresponds
            to the number of inversion times.
        """
        return super().__call__(m0, t1)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of InversionRecovery.

        Note: Do not use. Instead, call the instance of the Operator as operator(x)"""
        ndim = max(m0.ndim, t1.ndim)
        ti = unsqueeze_right(self.ti, ndim - self.ti.ndim + 1)  # leftmost is time
        signal = m0 * (1 - 2 * torch.exp(-(ti / t1)))
        return (signal,)
