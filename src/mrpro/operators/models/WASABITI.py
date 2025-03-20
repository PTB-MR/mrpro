"""WASABITI signal model for mapping of B0, B1 and T1."""

from collections.abc import Sequence

import torch
from torch import nn

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON


class WASABITI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """WASABITI signal model."""

    def __init__(
        self,
        offsets: torch.Tensor | float | Sequence[float],
        recovery_time: torch.Tensor | float | Sequence[float],
        rf_duration: float | torch.Tensor = 0.005,
        b1_nominal: float | torch.Tensor = 3.75e-6,
        gamma: float = GYROMAGNETIC_RATIO_PROTON,
    ) -> None:
        """Initialize WASABITI signal model for mapping of B0, B1 and T1 [SCH2023]_.

        Parameters
        ----------
        offsets
            frequency offsets [Hz] with shape `(offsets, ...)`
        recovery_time
            recovery time between offsets [s] with shape `(offsets, ...)`
        rf_duration
            RF pulse duration [s]
        b1_nominal
            nominal B1 amplitude [T]
        gamma
            gyromagnetic ratio [Hz/T]

        References
        ----------
        .. [SCH2023] Schuenke P, Zimmermann F, Kaspar K, Zaiss M, Kolbitsch C (2023) An Analytic Solution for the
           Modified WASABI Method: Application to Simultaneous B0, B1 and T1 Mapping and Correction of CEST MRI,
           Proceedings of the Annual Meeting of ISMRM
        """
        super().__init__()
        # offsets determines the device
        offsets_tensor = torch.as_tensor(offsets)
        self.offsets = nn.Parameter(offsets_tensor, requires_grad=offsets_tensor.requires_grad)
        recovery_time_tensor = torch.as_tensor(recovery_time)
        self.recovery_time = nn.Parameter(
            recovery_time_tensor.to(device=offsets_tensor.device), requires_grad=recovery_time_tensor.requires_grad
        )
        rf_duration_tensor = torch.as_tensor(rf_duration, device=offsets_tensor.device)
        self.rf_duration = nn.Parameter(rf_duration_tensor, requires_grad=rf_duration_tensor.requires_grad)
        b1_nominal_tensor = torch.as_tensor(b1_nominal, device=offsets_tensor.device)
        self.b1_nominal = nn.Parameter(b1_nominal_tensor, requires_grad=b1_nominal_tensor.requires_grad)
        self.gamma = gamma

    def forward(self, b0_shift: torch.Tensor, relative_b1: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply WASABITI signal model.

        Parameters
        ----------
        b0_shift
            B0 shift [Hz]
            with shape `(*other, coils, z, y, x)`
        relative_b1
            relative B1 amplitude
            with shape `(*other, coils, z, y, x)`
        t1
            longitudinal relaxation time T1 [s]
            with shape `(*other, coils, z, y, x)`

        Returns
        -------
            signal with shape `(offsets, *other, coils, z, y, x)`
        """
        ndim = max(b0_shift.ndim, relative_b1.ndim, t1.ndim)
        offsets = unsqueeze_right(self.offsets, ndim - self.offsets.ndim + 1)  # leftmost is offset
        recovery_time = unsqueeze_right(self.recovery_time, ndim - self.recovery_time.ndim + 1)  # leftmost is offset
        b1_nominal = unsqueeze_right(self.b1_nominal, ndim - self.b1_nominal.ndim)
        rf_duration = unsqueeze_right(self.rf_duration, ndim - self.rf_duration.ndim)

        b1 = b1_nominal * relative_b1
        offsets_shifted = offsets - b0_shift
        mz_initial = 1.0 - torch.exp(-recovery_time / t1)

        signal = mz_initial * (
            1
            - 2
            * (torch.pi * b1 * self.gamma * rf_duration) ** 2
            * torch.sinc(rf_duration * torch.sqrt((b1 * self.gamma) ** 2 + offsets_shifted**2)) ** 2
        )
        return (signal,)
