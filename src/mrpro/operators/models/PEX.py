"""Saturation recovery signal model for T1 mapping."""

from collections.abc import Sequence

import torch

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON, volt_to_sqrt_kwatt


class PexSimple(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for preparation based B1+ mapping (PEX)."""

    def __init__(
        self,
        voltages: float | torch.Tensor | Sequence[int],
        prep_delay: float | torch.Tensor,
        t1: float | torch.Tensor,
        pulse_duration: float | torch.Tensor,
        n_tx: int = 1,
    ) -> None:
        """Initialize preparation based B1+ mapping (PEX) signal model.

        Parameters
        ----------
        voltages
            voltages. Shape `(Voltages, ...)`.
        prep_delay
            preparation delay. Shape `(1, ...)`.
        t1
            longitudinal relaxation time T1. Shape `(1, ...)`.
        pulse_duration
            rect pulse duration in seconds. Shape `(1, ...)`.
        n_tx
            number of transmit channels.
        """
        super().__init__()
        voltages = torch.as_tensor(voltages) * torch.sqrt(torch.tensor(n_tx, dtype=torch.float))
        prep_delay = torch.as_tensor(prep_delay)
        t1 = torch.as_tensor(t1)
        pulse_duration = torch.as_tensor(pulse_duration)
        self.voltages = torch.nn.Parameter(voltages, requires_grad=voltages.requires_grad)
        self.prep_delay = torch.nn.Parameter(prep_delay, requires_grad=prep_delay.requires_grad)
        self.t1 = torch.nn.Parameter(t1, requires_grad=t1.requires_grad)
        self.pulse_duration = torch.nn.Parameter(pulse_duration, requires_grad=pulse_duration.requires_grad)

    def forward(self, a: torch.tensor) -> tuple[torch.Tensor,]:
        """Apply PEX signal model.

        Parameters
        ----------
        a
            parameter a in µT/sqrt(kW) translating voltage of the coil to flip angle
            with shape `(*other, coils, z, y, x)`

        Returns
        -------
            signal with shape `(voltage, *other, coils, z, y, x)`
        """
        ndim = a.ndim
        voltages = unsqueeze_right(self.voltages, ndim - self.voltages.ndim + 1)
        prep_delay = unsqueeze_right(self.prep_delay, ndim - self.prep_delay.ndim + 1)
        t1 = unsqueeze_right(self.t1, ndim - self.t1.ndim + 1)
        pulse_duration = unsqueeze_right(self.pulse_duration, ndim - self.pulse_duration.ndim + 1)

        # this is mainly cos(FA), where FA = gamma * a * voltage * t
        signal = 1 - (
            1
            - torch.cos(
                a * volt_to_sqrt_kwatt(voltages) * 1e-6 * pulse_duration * GYROMAGNETIC_RATIO_PROTON * 2 * torch.pi
            )
        ) * torch.exp(-prep_delay / t1)
        return (signal,)
