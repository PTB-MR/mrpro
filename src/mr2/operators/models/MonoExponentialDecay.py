"""Mono-exponential decay signal model."""

from collections.abc import Sequence

import torch

from mr2.operators.SignalModel import SignalModel
from mr2.utils.reshape import unsqueeze_right


class MonoExponentialDecay(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for mono-exponential decay."""

    def __init__(self, decay_time: float | torch.Tensor | Sequence[float]):
        """Initialize mono-exponential signal model.

        Can, for example, be used to model T2.

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
        :math:`S(t) = M_0 e^{-t / T}`,
        where `t` are the decay times and `T` is the decay constant.

        Parameters
        ----------
        m0
            Equilibrium signal / proton density.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        decay_constant
            Exponential decay constant (e.g., T2, T2*, T1rho).
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.

        Returns
        -------
            Signal calculated for each decay time.
            Shape `(times ...)`, for example `(times, *other, coils, z, y, x)`, or `(times, samples)`
            where `times` is the number of decay times.
        """
        return super().__call__(m0, decay_constant)

    def forward(self, m0: torch.Tensor, decay_constant: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of MonoExponentialDecay.

        .. note::
            Prefer calling the instance of the MonoExponentialDecay as ``operator(x)`` over directly calling this
            method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        ndim = max(m0.ndim, decay_constant.ndim)
        decay_time = unsqueeze_right(self.decay_time, ndim - self.decay_time.ndim + 1)  # leftmost is time
        signal = m0 * torch.exp(-(decay_time / decay_constant))
        return (signal,)
