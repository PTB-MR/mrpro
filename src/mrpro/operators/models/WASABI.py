"""WASABI signal model for mapping of B0 and B1."""

import torch
from torch import nn

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON


class WASABI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """WASABI signal model."""

    def __init__(
        self,
        offsets: torch.Tensor,
        rf_duration: float | torch.Tensor = 0.005,
        b1_nominal: float | torch.Tensor = 3.70e-6,
        gamma: float | torch.Tensor = GYROMAGNETIC_RATIO_PROTON,
    ) -> None:
        """Initialize WASABI signal model for mapping of B0 and B1 [SCHU2016]_.

        Parameters
        ----------
        offsets
            frequency offsets [Hz]
            with shape `(offsets, ...)`
        rf_duration
            RF pulse duration [s]
        b1_nominal
            nominal B1 amplitude [T]
        gamma
            gyromagnetic ratio [Hz/T]

        References
        ----------
        .. [SCHU2016] Schuenke P, Zaiss M (2016) Simultaneous mapping of water shift and B1(WASABI)—Application to
           field-Inhomogeneity correction of CEST MRI data. MRM 77(2). https://doi.org/10.1002/mrm.26133
        """
        super().__init__()

        rf_duration = torch.as_tensor(rf_duration)
        b1_nominal = torch.as_tensor(b1_nominal)
        gamma = torch.as_tensor(gamma)

        # nn.Parameters allow for grad calculation
        self.offsets = nn.Parameter(offsets, requires_grad=offsets.requires_grad)
        self.rf_duration = nn.Parameter(rf_duration, requires_grad=rf_duration.requires_grad)
        self.b1_nominal = nn.Parameter(b1_nominal, requires_grad=b1_nominal.requires_grad)
        self.gamma = gamma

    def forward(
        self,
        b0_shift: torch.Tensor,
        relative_b1: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Apply WASABI signal model.

        Parameters
        ----------
        b0_shift
            B0 shift [Hz]
            with shape `(*other, coils, z, y, x)`
        relative_b1
            relative B1 amplitude
            with shape `(*other, coils, z, y, x)`
        c
            additional fit parameter for the signal model
            with shape `(*other, coils, z, y, x)`
        d
            additional fit parameter for the signal model
            with shape `(*other, coils, z, y, x)`

        Returns
        -------
            signal with shape `(offsets, *other, coils, z, y, x)`
        """
        offsets = unsqueeze_right(self.offsets, b0_shift.ndim - (self.offsets.ndim - 1))  # -1 for offset
        offsets_shifted = offsets - b0_shift
        b1 = self.b1_nominal * relative_b1

        signal = (
            c
            - d
            * (torch.pi * b1 * self.gamma * self.rf_duration) ** 2
            * torch.sinc(self.rf_duration * torch.sqrt((b1 * self.gamma) ** 2 + offsets_shifted**2)) ** 2
        )
        return (signal,)
