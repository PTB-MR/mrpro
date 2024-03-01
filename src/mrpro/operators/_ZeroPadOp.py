"""Class for Zero Pad Operator."""

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

from mrpro.operators import LinearOperator
from mrpro.utils import zero_pad_or_crop


class ZeroPadOp(LinearOperator):
    """Zero Pad operator class."""

    def __init__(self, dim: tuple[int, ...], orig_shape: tuple[int, ...], padded_shape: tuple[int, ...]) -> None:
        """Zero Pad Operator class.

        The operator carries out zero-padding if the padded_shape is larger than orig_shape and cropping if the
        padded_shape is smaller.

        Parameters
        ----------
        dim
            dim along which padding should be applied
        orig_shape
            shape of original data along dim, same length as dim
        padded_shape
            shape of padded data along dim, same length as dim
        """
        if len(dim) != len(orig_shape) or len(dim) != len(padded_shape):
            raise ValueError('Dim, orig_shape and padded_shape have to be of same length')

        super().__init__()
        self.dim: tuple[int, ...] = dim
        self.orig_shape: tuple[int, ...] = orig_shape
        self.padded_shape: tuple[int, ...] = padded_shape

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Pad or crop data.

        Parameters
        ----------
        x
            data with shape orig_shape

        Returns
        -------
            data with shape padded_shape
        """
        return (zero_pad_or_crop(x, self.padded_shape, self.dim),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Crop or pad data.

        Parameters
        ----------
        x
            data with shape padded_shape

        Returns
        -------
            data with shape orig_shape
        """
        return (zero_pad_or_crop(x, self.orig_shape, self.dim),)
