"""Saturation recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils.reshape import unsqueeze_right


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

    def __call__(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the saturation recovery signal model.

        Calculates the signal based on the formula:
        S(t_sat) = M0 * (1 - exp(-t_sat / T1))
        where t_sat are the saturation times.

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
            Signal calculated for each saturation time.
            Shape `(time, *other, coils, z, y, x)`, where `time` corresponds
            to the number of saturation times.
        """
        return super().__call__(m0, t1)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of SaturationRecovery.

        Note: Do not use. Instead, call the instance of the Operator as operator(x)"""
        ndim = max(m0.ndim, t1.ndim)
        saturation_time = unsqueeze_right(
            self.saturation_time, ndim - self.saturation_time.ndim + 1
        )  # leftmost is time
        signal = m0 * (1 - torch.exp(-(saturation_time / t1)))
        return (signal,)
