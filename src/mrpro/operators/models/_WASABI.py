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

from mrpro.operators import SignalModel


class WASABI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
    """WASABI signal model."""

    def __init__(
        self,
        offsets: torch.Tensor,
        tp: float | torch.Tensor = 0.005,
        b1_nom: float | torch.Tensor = 3.70,
        gamma: float | torch.Tensor = 42.5764,
        freq: float | torch.Tensor = 127.7292,
    ) -> None:
        """Initialize WASABI signal model for mapping of B0 and B1.

        For more details see: https://doi.org/10.1002/mrm.26133

        Parameters
        ----------
        offsets
            frequency offsets [Hz], must be 1D tensor
        tp
            RF pulse duration [s], by default 0.005
        b1_nom
            nominal B1 amplitude [µT], by default 3.70
        gamma
            gyromagnetic ratio [MHz/T], by default 42.5764
        freq
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
        self.tp = nn.Parameter(tp, requires_grad=tp.requires_grad)
        self.b1_nom = nn.Parameter(b1_nom, requires_grad=b1_nom.requires_grad)
        self.gamma = nn.Parameter(gamma, requires_grad=gamma.requires_grad)
        self.freq = nn.Parameter(freq, requires_grad=freq.requires_grad)

    def forward(
        self,
        b0_shift: torch.Tensor,
        rb1: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Apply WASABI signal model.

        Parameters
        ----------
        b0_shift
            B0 shift [Hz]
        rb1
            relative B1 amplitude
        c
            additional fit parameter for the signal model
        d
            additional fit parameter for the signal model

        Returns
        -------
            signal with dimensions ((... offsets), coils, z, y, x)
        """
        offsets = self.offsets[(...,) + (None,) * b0_shift.ndim]
        delta_x = offsets - b0_shift
        b1 = self.b1_nom * rb1

        signal = (
            c
            - d
            * (torch.pi * b1 * self.gamma * self.tp) ** 2
            * torch.sinc(self.tp * torch.sqrt((b1 * self.gamma) ** 2 + delta_x**2)) ** 2
        )
        signal = rearrange(signal, 'offsets ... c z y x -> (... offsets) c z y x')
        return (signal,)
