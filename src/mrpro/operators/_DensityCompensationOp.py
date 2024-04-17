"""Class for Density Compensation Operator."""

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

from __future__ import annotations

import torch

from mrpro.data._DcfData import DcfData
from mrpro.operators._EinsumOp import EinsumOp


class DensityCompensationOp(EinsumOp):
    """Density Compensation Operator."""

    def __init__(self, dcf: DcfData | torch.Tensor) -> None:
        """Initialize a Density Compensation Operator.

        Parameters
        ----------
        dcf
           Density Compensation Data
        """
        if isinstance(dcf, DcfData):
            # only tensors can be used as buffers
            dcf = dcf.data
        super().__init__(dcf, '... k2 k1 k0 ,... coil k2 k1 k0 ->... coil k2 k1 k0')
