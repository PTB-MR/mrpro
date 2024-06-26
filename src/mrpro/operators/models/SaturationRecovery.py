"""Saturation recovery signal model for T1 mapping."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

from mrpro.operators.SignalModel import SignalModel


class SaturationRecovery(SignalModel[torch.Tensor, torch.Tensor]):
    """Signal model for saturation recovery."""

    def __init__(self, ti: float | torch.Tensor):
        """Initialize saturation recovery signal model for T1 mapping.

        Parameters
        ----------
        ti
            saturation times
            with shape (time, ...)
        """
        super().__init__()
        ti = torch.as_tensor(ti)
        self.ti = torch.nn.Parameter(ti, requires_grad=ti.requires_grad)

    def forward(self, m0: torch.Tensor, t1: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply saturation recovery signal model.

        Parameters
        ----------
        m0
            equilibrium signal / proton density
            with shape (... other, coils, z, y, x)
        t1
            longitudinal relaxation time T1
            with shape (... other, coils, z, y, x)

        Returns
        -------
            signal
            with shape (time ... other, coils, z, y, x)
        """
        delta_ndim = m0.ndim - (self.ti.ndim - 1)  # -1 for time
        ti = self.ti[..., *[None] * (delta_ndim)] if delta_ndim > 0 else self.ti
        signal = m0 * (1 - torch.exp(-(ti / t1)))
        return (signal,)
