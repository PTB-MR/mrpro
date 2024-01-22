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
    def __init__(self, ti: torch.Tensor):
        """Parameters needed for saturation recovery.

        Parameters
        ----------
        ti
            inversion times
        """
        super().__init__()
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
        """Apply the forward model.

        Parameters
        ----------
            Quantitative parameters m0, t1 with dimensions (other, c, z, y, x)

        Returns
        -------
            Image data tensor (other, c, z, y, x)
        """
        t1 = torch.where(t1 == 0, 1e-10, t1)
        ti = self.ti[(...,) + (None,) * (m0.ndim)]
        y = m0 * (1 - torch.exp(-(ti / t1)))
        res = rearrange(y, 't ... c z y x -> (... t) c z y x')
        return res
