"""Mono-exponential decay signal model."""

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right


class MonoExponentialDecay(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for mono-exponential decay."""

    def __init__(self, decay_time: float | torch.Tensor):
        """Initialize mono-exponential signal model.

        Parameters
        ----------
        decay_time
            time points when model is evaluated
            with shape (time, ...)
        """
        super().__init__()
        decay_time = torch.as_tensor(decay_time)
        self.decay_time = torch.nn.Parameter(decay_time, requires_grad=decay_time.requires_grad)

    def forward(self, m0: torch.Tensor, decay_constant: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply mono-exponential signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        decay_constant
            exponential decay constant (e.g. T2, T2* or T1rho)
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal with shape (time ... other, coils, z, y, x)
        """
        decay_time = unsqueeze_right(self.decay_time, m0.ndim - (self.decay_time.ndim - 1))  # -1 for time
        signal = m0 * torch.exp(-(decay_time / decay_constant))
        return (signal,)
