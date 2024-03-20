# Copyright 2023 Physikalisch-Technische Bundesanstalt
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
from einops import rearrange

from mrpro.operators import SignalModel


class MOLLI(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Signal model for Modified Look-Locker inversion recovery (MOLLI)."""

    def __init__(self, ti: torch.Tensor):
        """Initialize MOLLI signal model for T1 mapping.

        Parameters
        ----------
        ti
            inversion times [s]
        """
        super().__init__()
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, a: torch.Tensor, b: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply MOLLI signal model.

        Parameters
        ----------
        a
            Parameter a in MOLLI signal model
        b
            Parameter b in MOLLI signal model
        t1
            longitudinal relaxation time T1 [s]

        Returns
        -------
            signal with dimensions ((... inv_times), coils, z, y, x)
        """
        ti = self.ti[..., None, :, None, None, None, None]  # *other t=1, c, z, y, x
        a = a.unsqueeze(-5)  # *other t=1, c, z, y, x
        b = b.unsqueeze(-5)
        t1 = t1.unsqueeze(-5)
        c = b / torch.where(a == 0, 1e-10, a)
        t1 = torch.where(t1 == 0, t1 + 1e-10, t1)
        signal = a * (1 - c * torch.exp(ti / t1 * (1 - c)))  # *other t=len(ti), c, z, y, x
        signal = rearrange(signal, '... other t c z y x -> ... (other t) c z y x')
        return (signal,)
