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


class WASABITI(Operator):
    def __init__(
        self,
        offsets: torch.Tensor,
        trec: torch.Tensor,
        tp: float = 0.005,
        b1_nom: float = 3.75,
        gamma: float = 42.5764,
        freq: float = 127.7292,
    ) -> None:
        """WASABITI function for simultaneous determination of B1, B0 and T1.

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
        self.offsets = offsets
        self.trec = trec
        self.tp = tp
        self.b1_nom = b1_nom
        self.gamma = gamma
        self.freq = freq

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        b0_shift = qdata[0].unsqueeze(0)
        rb1 = qdata[1].unsqueeze(0)
        t1 = qdata[2].unsqueeze(0)

        b1 = self.b1_nom * rb1

        # ensure correct dimensionality
        offsets = self.offsets[(...,) + (None,) * qdata[0].ndim]
        trec = self.trec[(...,) + (None,) * qdata[0].ndim]

        da = offsets - b0_shift
        Mzi = 1.0 - torch.exp(torch.multiply(-1.0 / t1, trec))

        res = Mzi * (
            1
            - 2
            * (torch.pi * b1 * self.gamma * self.tp) ** 2
            * torch.sinc(self.tp * torch.sqrt((b1 * self.gamma) ** 2 + da**2)) ** 2
        )

        # c = coils, ... = other (may be multi-dimensional)
        return rearrange(res, 'offset ... c z y x -> (... offset) c z y x')
