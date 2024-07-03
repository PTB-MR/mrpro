"""WASABI signal model for mapping of B0 and B1."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn

from mrpro.operators.SignalModel import SignalModel


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
            frequency offsets [Hz]
            with shape (offsets, ...)
        tp
            RF pulse duration [s]
        b1_nom
            nominal B1 amplitude [µT]
        gamma
            gyromagnetic ratio [MHz/T]
        freq
            larmor frequency [MHz]
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
        relative_b1: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> tuple[torch.Tensor,]:
        """Apply WASABI signal model.

        Parameters
        ----------
        b0_shift
            B0 shift [Hz]
            with shape (... other, coils, z, y, x)
        relative_b1
            relative B1 amplitude
            with shape (... other, coils, z, y, x)
        c
            additional fit parameter for the signal model
            with shape (... other, coils, z, y, x)
        d
            additional fit parameter for the signal model
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape (offsets ... other, coils, z, y, x)
        """
        delta_ndim = b0_shift.ndim - (self.offsets.ndim - 1)  # -1 for offset
        offsets = self.offsets[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.offsets
        delta_x = offsets - b0_shift
        b1 = self.b1_nom * relative_b1

        signal = (
            c
            - d
            * (torch.pi * b1 * self.gamma * self.tp) ** 2
            * torch.sinc(self.tp * torch.sqrt((b1 * self.gamma) ** 2 + delta_x**2)) ** 2
        )
        return (signal,)
