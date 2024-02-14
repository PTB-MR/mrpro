"""Squared L2-norm."""

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
import torch.nn.functional as F

from mrpro.operators import Operator


class L2_data_discrepancy(Operator):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()
        """Squared L2-norm loss function, i.e.
            || . - data ||_2^2

        Parameters
        ----------
        data
            observed data
        """

        # observed data
        self.data = torch.view_as_real(data) if torch.is_complex(data) else data

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = torch.view_as_real(x) if torch.is_complex(x) else x
        return (F.mse_loss(x, self.data),)
