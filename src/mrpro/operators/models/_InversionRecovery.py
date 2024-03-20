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


class InversionRecovery(SignalModel[torch.Tensor, torch.Tensor]):
    """Inversion Recovery signal model."""

    def __init__(self, ti: torch.Tensor):
        """Initialize Inversion Recovery signal model for T1 mapping.

        Parameters
        ----------
        ti
            inversion times [s]
        """
        super().__init__()
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply Inversion Recovery signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape ... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1
            with shape ... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape ... (other inv_times), coils, z, y, x)
        """
        ti = self.ti[..., None, :, None, None, None, None]  # *other t, c, z, y, x
        m0 = m0.unsqueeze(-5)  # *other t, c, z, y, x
        t1 = t1.unsqueeze(-5)  # *other t, c, z, y, x
        signal = m0 * (1 - 2 * torch.exp(-(ti / t1)))
        signal = rearrange(signal, '... other t c z y x -> ... ( other t ) c z y x')
        return (signal,)
