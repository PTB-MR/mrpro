"""Class for Sensitivity Operator."""

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

from mrpro.data import CsmData
from mrpro.operators import LinearOperator


class SensitivityOp(LinearOperator):
    """Sensitivity operator class."""

    def __init__(self, csm: CsmData):
        self.C = csm.data

    def forward(self, img_data: torch.Tensor):
        """Apply the forward operator, thus expand the coils dimension.

        Parameters
        ----------
        img_data
            image data tensor with dimensions (all_other, 1, z, y, x).

        Returns
        -------
        torch.Tensor
            image data tensor with dimensions (all_other, coils, z, y, x).
        """
        return self.C * img_data

    def adjoint(self, img_data: torch.Tensor):
        """Apply the adjoint operator, thus reduce the coils dimension.

        Parameters
        ----------
        img_data
            image data tensor with dimensions (all_other, coils, z, y, x).

        Returns
        -------
        torch.Tensor
            image data tensor with dimensions (all_other, 1, z, y, x).
        """

        return (self.C.conj() * img_data).sum(-4, keepdim=True)
