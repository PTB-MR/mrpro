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


class Molli(Operator):
    def __init__(self, ti: list[float], n: int, rr: float):
        """Parameters needed to compute t = ti + (n - 1) * rr.

        Parameters
        ----------
        ti
            inversion times.
        n
            image number within the Look-Locker experiment
        rr
            heartbeat interval

        """
        self.ti = ti
        self.n = n
        self.rr = rr

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        """Apply the forward model.

        Parameters
        ----------
        qdata
            Quantitative parameter tensor (params, other, c, z, y, x)
            params: (A, B, T1)

        Returns
        -------
            Image data tensor (other, c, z, y, x)
        """
        A = qdata[0].unsqueeze(0)
        B = qdata[1].unsqueeze(0)
        T1 = qdata[2].unsqueeze(0)
        t = torch.Tensor(self.ti) + (self.n - 1) * self.rr
        t = rearrange(t, 't -> t 1 1 1 1 1')
        T1_star = torch.div(T1, (torch.div(B, A) - 1))
        y = A - B * torch.exp(-torch.div(t, T1_star))
        res = rearrange(y, 't other c z y x -> (t other) c z y x')
        return res
