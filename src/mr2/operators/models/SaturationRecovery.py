"""Saturation recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mr2.operators.SignalModel import SignalModel
from mr2.utils.reshape import unsqueeze_right


class SaturationRecovery(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for saturation recovery."""

    def __init__(self, saturation_time: float | torch.Tensor | Sequence[float]) -> None:
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
        :math:`S(t_{sat}) = M_0 (1 - e^{-t_{sat} / T_1})`,
        where `t_{sat}` are the saturation times.

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
            Signal calculated for each saturation time.
            Shape `(times ...)`, for example `(times, *other, coils, z, y, x)`, or `(times, samples)`
            where `times` is the number of saturation times.
        """
        return super().__call__(m0, t1)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of SaturationRecovery.

        .. note::
            Prefer calling the instance of the SaturationRecovery as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        ndim = max(m0.ndim, t1.ndim)
        saturation_time = unsqueeze_right(
            self.saturation_time, ndim - self.saturation_time.ndim + 1
        )  # leftmost is time
        signal = m0 * (1 - torch.exp(-(saturation_time / t1)))
        return (signal,)
