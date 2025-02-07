"""Saturation recovery signal model for T1 mapping."""

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right, unsqueeze_tensors_right


class SaturationRecovery(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for saturation recovery."""

    def __init__(self, ti: float | torch.Tensor):
        """Initialize saturation recovery signal model for T1 mapping.

        Parameters
        ----------
        ti
            saturation times
            with shape (time, ...)
        """
        super().__init__()
        ti = torch.as_tensor(ti)
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply saturation recovery signal model.

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
        m0, t1 = unsqueeze_tensors_right(m0, t1)
        ti = unsqueeze_right(self.ti, m0.ndim - self.ti.ndim + 1)  # leftmost is time
        signal = m0 * (1 - torch.exp(-(ti / t1)))
        return (signal,)
