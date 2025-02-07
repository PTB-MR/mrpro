"""WASABITI signal model for mapping of B0, B1 and T1."""

import torch
from torch import nn

from mrpro.operators.SignalModel import SignalModel
from mrpro.utils import unsqueeze_right, unsqueeze_tensors_right
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON


class WASABITI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """WASABITI signal model."""

    def __init__(
        self,
        offsets: torch.Tensor,
        recovery_time: torch.Tensor,
        rf_duration: float | torch.Tensor = 0.005,
        b1_nominal: float | torch.Tensor = 3.75e-6,
        gamma: float | torch.Tensor = GYROMAGNETIC_RATIO_PROTON,
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
        # convert all parameters to tensors
        rf_duration = torch.as_tensor(rf_duration)
        b1_nominal = torch.as_tensor(b1_nominal)
        gamma = torch.as_tensor(gamma)

        if recovery_time.shape != offsets.shape:
            raise ValueError(
                f'Shape of recovery_time ({recovery_time.shape}) and offsets ({offsets.shape}) needs to be the same.'
            )

        # nn.Parameters allow for grad calculation
        self.offsets = nn.Parameter(offsets, requires_grad=offsets.requires_grad)
        self.recovery_time = nn.Parameter(recovery_time, requires_grad=recovery_time.requires_grad)
        self.rf_duration = nn.Parameter(rf_duration, requires_grad=rf_duration.requires_grad)
        self.b1_nominal = nn.Parameter(b1_nominal, requires_grad=b1_nominal.requires_grad)
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
        b0_shift, relative_b1, t1 = unsqueeze_tensors_right(b0_shift, relative_b1, t1)
        delta_ndim = b0_shift.ndim - self.offsets.ndim + 1  # leftmost is offset
        offsets = unsqueeze_right(self.offsets, delta_ndim)
        recovery_time = unsqueeze_right(self.recovery_time, delta_ndim)

        b1 = self.b1_nominal * relative_b1
        offsets_shifted = offsets - b0_shift
        mz_initial = 1.0 - torch.exp(-recovery_time / t1)

        signal = mz_initial * (
            1
            - 2
            * (torch.pi * b1 * self.gamma * self.rf_duration) ** 2
            * torch.sinc(self.rf_duration * torch.sqrt((b1 * self.gamma) ** 2 + offsets_shifted**2)) ** 2
        )
        return (signal,)
