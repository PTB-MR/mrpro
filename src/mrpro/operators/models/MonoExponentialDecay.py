"""Mono-exponential decay signal model."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils.reshape import unsqueeze_right


class MonoExponentialDecay(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for mono-exponential decay."""

    def __init__(self, decay_time: float | torch.Tensor | Sequence[float]):
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

    def __call__(self, m0: torch.Tensor, decay_constant: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the mono-exponential decay signal model.

        Calculates the signal based on the formula:
        S(t) = M0 * exp(-t / T)
        where t are the decay times and T is the decay constant.

        Parameters
        ----------
        m0
            Equilibrium signal or proton density.
            Expected shape `(*other, coils, z, y, x)`.
        decay_constant
            Exponential decay constant (e.g., T2, T2*, T1rho).
            Expected shape `(*other, coils, z, y, x)`.

        Returns
        -------
        tuple[torch.Tensor,]
            Signal calculated for each decay time.
            Shape `(time, *other, coils, z, y, x)`, where `time` corresponds
            to the number of decay times.
        """
        return super().__call__(m0, decay_constant)

    def forward(self, m0: torch.Tensor, decay_constant: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of MonoExponentialDecay.

        Note: Do not use. Instead, call the instance of the Operator as operator(x)"""
        ndim = max(m0.ndim, decay_constant.ndim)
        decay_time = unsqueeze_right(self.decay_time, ndim - self.decay_time.ndim + 1)  # leftmost is time
        signal = m0 * torch.exp(-(decay_time / decay_constant))
        return (signal,)
