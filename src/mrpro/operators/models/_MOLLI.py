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


class MOLLI(Operator):
    def __init__(self, ti: torch.Tensor):
        """Parameters needed for MOLLI.

        Parameters
        ----------
        ti
            inversion times
        """
        super().__init__()
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        """Apply the forward model.

        Parameters
        ----------
        qdata
            Quantitative parameter tensor (params, other, c, z, y, x)
            params: (a, b, t1)

        Returns
        -------
            Image data tensor (other, c, z, y, x)
        """
        a, b, t1 = qdata.unsqueeze(1)
        t1 = torch.where(t1 == 0, 1e-10, t1)
        a = torch.where(a == 0 | torch.equal((b / a), torch.ones_like(a)), a + 1e-10, a)
        ti = self.ti[(...,) + (None,) * (qdata[0].ndim)]
        t1_star = t1 / ((b / a) - 1)
        y = a - b * torch.exp(-(ti / (t1_star)))
        res = rearrange(y, 't ... c z y x -> (... t) c z y x')
        return res
