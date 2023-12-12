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

from mrpro.operators import Operator


class WASABI(Operator):
    def __init__(
        self,
        offsets: torch.Tensor,
        tp: float = 0.005,
        b1_nom: float = 3.70,
        gamma: float = 42.5764,
        freq: float = 127.7292,
    ) -> None:
        """WASABI function for simultaneous determination of B1 and B0.

        For more details see: https://doi.org/10.1002/mrm.26133

        Parameters
        ----------
        offsets
            frequency offsets [Hz], must be 1D tensor
        tp, optional
            RF pulse duration [s], by default 0.005
        b1_nom, optional
            nominal B1 amplitude [µT], by default 3.70
        gamma, optional
            gyromagnetic ratio [MHz/T], by default 42.5764
        freq, optional
            larmor frequency [MHz], by default 127.7292
        """

        self.offsets = offsets
        self.tp = tp
        self.b1_nom = b1_nom
        self.gamma = gamma
        self.freq = freq

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        b0_shift = qdata[0].unsqueeze(0)
        rb1 = qdata[1].unsqueeze(0)
        c = qdata[2].unsqueeze(0)
        d = qdata[3].unsqueeze(0)

        # ensure correct dimensionality
        offsets = self.offsets[(...,) + (None,) * qdata[0].ndim]
        delta_x = offsets - b0_shift
        b1 = self.b1_nom * rb1

        res = (
            c
            - d
            * (torch.pi * b1 * self.gamma * self.tp) ** 2
            * torch.sinc(self.tp * torch.sqrt((b1 * self.gamma) ** 2 + delta_x**2)) ** 2
        )

        # c = coils, ... = other (may be multi-dimensional)
        return rearrange(res, 'offset ... c z y x -> (... offset) c z y x')
