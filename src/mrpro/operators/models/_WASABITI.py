# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import torch
from einops import rearrange
from torch import nn

from mrpro.operators._SignalModel import SignalModel


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
        """Initialize WASABITI signal model for mapping of B0, B1 and T1.

        For more details see: Proc. Intl. Soc. Mag. Reson. Med. 31 (2023): 0906

        Parameters
        ----------
        offsets
            frequency offsets [Hz]
        trec
            recovery time between offsets [s]
        tp, optional
            RF pulse duration [s], by default 0.005
        b1_nom, optional
            nominal B1 amplitude [µT], by default 3.75
        gamma, optional
            gyromagnetic ratio [MHz/T], by default 42.5764
        freq, optional
            larmor frequency [MHz], by default 127.7292
        """
        super().__init__()
        # convert all parameters to tensors
        tp = torch.as_tensor(tp)
        b1_nom = torch.as_tensor(b1_nom)
        gamma = torch.as_tensor(gamma)
        freq = torch.as_tensor(freq)

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
        rb1
            relative B1 amplitude
        t1
            longitudinal relaxation time T1 [s]

        Returns
        -------
            signal with dimensions ((... offsets), coils, z, y, x)
        """
        b1 = self.b1_nom * rb1

        # ensure correct dimensionality
        offsets = self.offsets[(...,) + (None,) * b0_shift.ndim]
        trec = self.trec[(...,) + (None,) * b0_shift.ndim]

        da = offsets - b0_shift
        mz_initial = 1.0 - torch.exp(torch.multiply(-1.0 / t1, trec))

        signal = mz_initial * (
            1
            - 2
            * (torch.pi * b1 * self.gamma * self.tp) ** 2
            * torch.sinc(self.tp * torch.sqrt((b1 * self.gamma) ** 2 + da**2)) ** 2
        )

        signal = rearrange(signal, 'offset ... c z y x -> (... offset) c z y x')
        return (signal,)
