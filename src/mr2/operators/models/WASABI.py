"""WASABI signal model for mapping of B0 and B1."""

from collections.abc import Sequence

import torch
from torch import nn

from mr2.operators.SignalModel import SignalModel
from mr2.utils.reshape import unsqueeze_right
from mr2.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON


class WASABI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """WASABI signal model."""

    def __init__(
        self,
        offsets: torch.Tensor | Sequence[float] | float,
        rf_duration: float | torch.Tensor = 0.005,
        b1_nominal: float | torch.Tensor = 3.70e-6,
        gamma: float = GYROMAGNETIC_RATIO_PROTON,
    ) -> None:
        """Initialize WASABI signal model for mapping of B0 and B1 [SCHU2016]_.

        This model uses a slight modification from the original published model.
        The parameter `a` corresponds to `d/c` in the original model.

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
        # offsets determines the device
        offsets_tensor = torch.as_tensor(offsets)
        self.offsets = nn.Parameter(offsets_tensor, requires_grad=offsets_tensor.requires_grad)
        rf_duration_tensor = torch.as_tensor(rf_duration, device=offsets_tensor.device)
        self.rf_duration = nn.Parameter(rf_duration_tensor, requires_grad=rf_duration_tensor.requires_grad)
        b1_nominal_tensor = torch.as_tensor(b1_nominal, device=offsets_tensor.device)
        self.b1_nominal = nn.Parameter(b1_nominal_tensor, requires_grad=b1_nominal_tensor.requires_grad)
        self.gamma = gamma

    def __call__(
        self,
        b0_shift: torch.Tensor,
        relative_b1: torch.Tensor,
        c: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Apply the WASABI (Water Shift and B1) signal model.

        Parameters
        ----------
        b0_shift
            B0 field in homogeneity or off-resonance shift in Hz.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        relative_b1
            Relative B1 amplitude scaling factor (actual B1 / nominal B1).
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        c
            Signal amplitude parameter (related to M0).
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.
        a
            Signal modulation scaling parameter, corresponds to `d/c` in the original model.
            Shape `(...)`, for example `(*other, coils, z, y, x)` or `(samples)`.

        Returns
        -------
            Signal calculated for each frequency offset.
            Shape `(offsets ...)`, for example `(offsets, *other, coils, z, y, x)`, or `(offsets, samples)`
            where `offsets` is the number of frequency offsets.
        """
        return super().__call__(b0_shift, relative_b1, c, a)

    def forward(
        self,
        b0_shift: torch.Tensor,
        relative_b1: torch.Tensor,
        c: torch.Tensor,
        a: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Apply forward of WASABI.

        .. note::
            Prefer calling the instance of the WASABI as ``operator(x)`` over directly calling this method.
            See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        ndim = max(b0_shift.ndim, relative_b1.ndim, c.ndim, a.ndim)
        offsets = unsqueeze_right(self.offsets, ndim - self.offsets.ndim + 1)  # leftmost is offsets
        rf_duration = unsqueeze_right(self.rf_duration, ndim - self.rf_duration.ndim)
        b1_nominal = unsqueeze_right(self.b1_nominal, ndim - self.b1_nominal.ndim)

        offsets_shifted = offsets - b0_shift
        b1 = b1_nominal * relative_b1

        signal = c * (
            1
            - a
            * (torch.pi * b1 * self.gamma * rf_duration) ** 2
            * torch.sinc(rf_duration * torch.sqrt((b1 * self.gamma) ** 2 + offsets_shifted**2)) ** 2
        )
        return (signal,)
