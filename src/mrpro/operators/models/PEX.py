"""Saturation recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON, volt_to_sqrt_kwatt


class PEX(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for preparation based B1+ mapping (PEX)."""

    def __init__(
        self,
        voltages: float | torch.Tensor | Sequence[float],
        prep_delay: float | torch.Tensor,
        pulse_duration: float | torch.Tensor,
        n_tx: int = 1,
    ) -> None:
        """Initialize preparation based B1+ mapping (PEX) signal model.

        Parameters
        ----------
        voltages
            voltages. Shape `(Voltages, ...)`.
        prep_delay
            preparation delay. Shape `(...)`.
        pulse_duration
            rect pulse duration in seconds. Shape `(...)`.
        n_tx
            number of transmit channels.
        """
        super().__init__()
        voltages_ = torch.as_tensor(voltages) * n_tx**0.5
        prep_delay_ = torch.as_tensor(prep_delay)
        pulse_duration_ = torch.as_tensor(pulse_duration)
        self.voltages = torch.nn.Parameter(voltages_, requires_grad=voltages_.requires_grad)
        self.prep_delay = torch.nn.Parameter(prep_delay_, requires_grad=prep_delay_.requires_grad)
        self.pulse_duration = torch.nn.Parameter(pulse_duration_, requires_grad=pulse_duration_.requires_grad)

    def __call__(self, a: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply PEX signal model.

        Parameters
        ----------
        a
            Parameter a in µT/sqrt(kW) translating voltage of the coil to flip angle
            with shape `(...)`.
        t1
            Longitudinal relaxation time.

        Returns
        -------
            signal with shape `(voltage, *other, coils, z, y, x)`
        """
        return super().__call__(a, t1)

    def forward(self, a: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply PEX signal model.

        .. note::
            Prefer calling the instance of the PEX operator as ``operator(a, t1)`` over
            directly calling this method.
        """
        ndim = a.ndim
        voltages = unsqueeze_right(self.voltages, ndim - self.voltages.ndim + 1)  # +1 are voltages
        prep_delay = unsqueeze_right(self.prep_delay, ndim - self.prep_delay.ndim)
        t1 = unsqueeze_right(t1, ndim - t1.ndim)
        pulse_duration = unsqueeze_right(self.pulse_duration, ndim - self.pulse_duration.ndim)

        # this is mainly cos(FA), where FA = gamma * a * voltage * t
        signal = 1 - (
            1
            - torch.cos(
                a * volt_to_sqrt_kwatt(voltages) * 1e-6 * pulse_duration * GYROMAGNETIC_RATIO_PROTON * 2 * torch.pi
            )
        ) * torch.exp(-prep_delay / t1)
        return (signal,)
