"""Inversion recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mr2.operators.SignalModel import SignalModel
from mr2.utils.reshape import unsqueeze_right


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
        :math:`S(TI) = M_0 (1 - 2 * e^{-TI / T_1})`,
        where `TI` are the inversion times.

        Parameters
        ----------
        m0
            Equilibrium signal / proton density.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        t1
            Longitudinal relaxation time T1.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.

        Returns
        -------
            Signal calculated for each inversion time.
            Shape `(times ...)`, for example `(times, *other, coils, z, y, x)`, or `(times, samples)`
            where `times` is the number of inversion times.
        """
        return super().__call__(m0, t1)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of InversionRecovery.

        .. note::
            Prefer calling the instance of the InversionRecovery as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        ndim = max(m0.ndim, t1.ndim)
        ti = unsqueeze_right(self.ti, ndim - self.ti.ndim + 1)  # leftmost is time
        signal = m0 * (1 - 2 * torch.exp(-(ti / t1)))
        return (signal,)
