"""Mean squared error (MSE) data-discrepancy function."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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


class mse_data_discrepancy(Operator[torch.Tensor, tuple[torch.Tensor, ...]]):
    """MSE loss function.

        This class implements the function
            1./N * || . - data ||_2^2,
        where N equals to the number of elements of the tensor.

        N.B. if one of data or input is complex-valued, we
        identify the space C^N with R^2N and multiply the output
        by 2. By this, we achieve that for example
            MSE(1) = MES(1+1j*0) = 1.

    Parameters
    ----------
    data
        observed data
    """

    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()

        # observed data
        self.data = data

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Calculate the MSE of the input.

        Parameters
        ----------
        x
            tensor whose mse with respect to data should be calculated

        Returns
        -------
            MSE of the different of the input and the given data
        """
        if torch.is_complex(x) or torch.is_complex(self.data):
            factor = 2.0
            x = torch.view_as_real(x) if torch.is_complex(x) else torch.view_as_real(x + 1j * 0)
            data = (
                torch.view_as_real(self.data) if torch.is_complex(self.data) else torch.view_as_real(self.data + 1j * 0)
            )
        else:
            factor = 1.0
            data = self.data

        return (factor * F.mse_loss(x, data),)
