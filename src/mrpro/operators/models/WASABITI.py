"""WASABITI signal model for mapping of B0, B1 and T1."""

import torch
from torch import nn

from mrpro.operators.SignalModel import SignalModel


class WASABITI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """WASABITI signal model."""

    def __init__(
        self,
        offsets: torch.Tensor,
        trec: torch.Tensor,
        tp: float | torch.Tensor = 0.005,
        b1_nom: float | torch.Tensor = 3.75,
        gamma: float | torch.Tensor = 42.5764,
        freq: float | torch.Tensor = 127.7292,
    ) -> None:
        """Initialize WASABITI signal model for mapping of B0, B1 and T1 [PAP2023]_.

        Parameters
        ----------
        offsets
            frequency offsets [Hz] with shape (offsets, ...)
        trec
            recovery time between offsets [s] with shape (offsets, ...)
        tp
            RF pulse duration [s]
        b1_nom
            nominal B1 amplitude [µT]
        gamma
            gyromagnetic ratio [MHz/T]
        freq
            larmor frequency [MHz]

        References
        ----------
        .. [PAP2023] Papageorgakis C, Casagranda S (2023) Fast WASABI post-processing: Access to rapid B0 and B1
           correction in clinical routine for CEST MRI. MRM 102. https://doi.org/10.1016/j.mri.2023.06.001
        """
        super().__init__()
        # convert all parameters to tensors
        tp = torch.as_tensor(tp)
        b1_nom = torch.as_tensor(b1_nom)
        gamma = torch.as_tensor(gamma)
        freq = torch.as_tensor(freq)

        if trec.shape != offsets.shape:
            raise ValueError(f'Shape of trec ({trec.shape}) and offsets ({offsets.shape}) needs to be the same.')

        # nn.Parameters allow for grad calculation
        self.offsets = nn.Parameter(offsets, requires_grad=offsets.requires_grad)
        self.trec = nn.Parameter(trec, requires_grad=trec.requires_grad)
        self.tp = nn.Parameter(tp, requires_grad=tp.requires_grad)
        self.b1_nom = nn.Parameter(b1_nom, requires_grad=b1_nom.requires_grad)
        self.gamma = nn.Parameter(gamma, requires_grad=gamma.requires_grad)
        self.freq = nn.Parameter(freq, requires_grad=freq.requires_grad)

    def forward(self, b0_shift: torch.Tensor, rb1: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply WASABITI signal model.

        Parameters
        ----------
        b0_shift
            B0 shift [Hz]
            with shape (... other, coils, z, y, x)
        rb1
            relative B1 amplitude
            with shape (... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1 [s]
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal with shape (offsets ... other, coils, z, y, x)
        """
        delta_ndim = b0_shift.ndim - (self.offsets.ndim - 1)  # -1 for offset
        offsets = self.offsets[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.offsets
        trec = self.trec[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.trec

        b1 = self.b1_nom * rb1
        da = offsets - b0_shift
        mz_initial = 1.0 - torch.exp(-trec / t1)

        signal = mz_initial * (
            1
            - 2
            * (torch.pi * b1 * self.gamma * self.tp) ** 2
            * torch.sinc(self.tp * torch.sqrt((b1 * self.gamma) ** 2 + da**2)) ** 2
        )
        return (signal,)
