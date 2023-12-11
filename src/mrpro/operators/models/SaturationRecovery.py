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


class SaturationRecovery(Operator):
    def __init__(self, ti: list[float]):
        self.ti = ti

    def forward(self, qdata: torch.Tensor) -> torch.Tensor:
        """Apply the forward model.

        Parameters
        ----------
        qdata
            Quantitative parameter tensor (params, other, c, z, y, x)
            params: (M0, T1)

        Returns
        -------
            Image data tensor (other, c, z, y, x)
        """
        M0 = qdata[0].unsqueeze(0)
        T1 = qdata[1].unsqueeze(0)
        ti = rearrange(torch.Tensor(self.ti), 't -> t 1 1 1 1 1')
        y = M0 * (1 - torch.exp(-torch.div(ti, T1)))
        res = rearrange(y, 't other c z y x -> (t other) c z y x')
        return res
