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

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply inversion recovery signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape `(*other, coils, z, y, x)`
        t1
            longitudinal relaxation time T1
            with shape `(*other, coils, z, y, x)`

        Returns
        -------
            signal with shape `(time, *other, coils, z, y, x)`
        """
        ndim = max(m0.ndim, t1.ndim)
        ti = unsqueeze_right(self.ti, ndim - self.ti.ndim + 1)  # leftmost is time
        signal = m0 * (1 - 2 * torch.exp(-(ti / t1)))
        return (signal,)
