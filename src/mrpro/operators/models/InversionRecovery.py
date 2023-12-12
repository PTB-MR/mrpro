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


class InversionRecovery(Operator):
    def __init__(self, ti: torch.Tensor):
        super().__init__()
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        """Apply the forward model.

        Parameters
        ----------
        qdata
            Quantitative parameter tensor (params, other, c, z, y, x)
            params: (m0, t1)

        Returns
        -------
            Image data tensor (other, c, z, y, x)
        """
        m0, t1 = qdata.unsqueeze(1)
        ti = self.ti[(...,) + (None,) * (qdata[0].ndim)]
        y = m0 * (1 - 2 * torch.exp(-ti / (t1 + 1e-10)))
        res = rearrange(y, 't ... c z y x -> (... t) c z y x')
        return res
