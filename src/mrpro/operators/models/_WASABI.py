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
import torch.nn as nn
from einops import rearrange

from mrpro.operators import Operator


class WASABI(Operator):
    def __init__(
        self,
        offsets: torch.Tensor,
        tp: torch.Tensor = torch.Tensor([0.005]),
        b1_nom: torch.Tensor = torch.Tensor([3.70]),
        gamma: torch.Tensor = torch.Tensor([42.5764]),
        freq: torch.Tensor = torch.Tensor([127.7292]),
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
        super().__init__()
        # nn.Parameters allow for grad calculation
        self.offsets = nn.Parameter(offsets, requires_grad=offsets.requires_grad)
        self.tp = nn.Parameter(tp, requires_grad=tp.requires_grad)
        self.b1_nom = nn.Parameter(b1_nom, requires_grad=b1_nom.requires_grad)
        self.gamma = nn.Parameter(gamma, requires_grad=gamma.requires_grad)
        self.freq = nn.Parameter(freq, requires_grad=freq.requires_grad)

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        b0_shift, rb1, c, d = qdata.unsqueeze(1)

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


# TODO: do singal model methods also for complex variants
