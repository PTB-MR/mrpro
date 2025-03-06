"""Saturation recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right


class SaturationRecovery(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for saturation recovery."""

    def __init__(self, saturation_time: float | torch.Tensor | Sequence[int]) -> None:
        """Initialize saturation recovery signal model for T1 mapping.

        Parameters
        ----------
        saturation_time
            delay between saturation and acquisition. Shape `(time, ...)`.
        """
        super().__init__()
        saturation_time = torch.as_tensor(saturation_time)
        self.saturation_time = torch.nn.Parameter(saturation_time, requires_grad=saturation_time.requires_grad)

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
        ndim = max(m0.ndim, t1.ndim)
        saturation_time = unsqueeze_right(
            self.saturation_time, ndim - self.saturation_time.ndim + 1
        )  # leftmost is time
        signal = m0 * (1 - torch.exp(-(saturation_time / t1)))
        return (signal,)
