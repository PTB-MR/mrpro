"""Class for Pad Operator."""

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
from mrpro.utils import change_data_shape


class PadOp(LinearOperator):
    """Pad operator class."""

    def __init__(
        self,
        dim: tuple[int, ...] = (-3, -2, -1),
        orig_shape: tuple[int, ...] | None = None,
        padded_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Pad Operator class.

        The operator carries out zero-padding if the padded_shape is larger than orig_shape and cropping if the
        padded_shape is smaller.

        Parameters
        ----------
        dim, optional
            dim along which padding should be applied, by default last three dimensions (-1, -2, -3)
        orig_shape, optional
            shape of original data along dim
        padded_shape, optional
            shape of padded data along dim
        """
        self.dim: tuple[int, ...] = dim
        self.orig_shape: tuple[int, ...] | None = orig_shape
        self.padded_shape: tuple[int, ...] | None = padded_shape

    @staticmethod
    def _pad_data(x: torch.Tensor, dim: tuple[int, ...], padded_shape: tuple[int, ...] | None) -> torch.Tensor:
        """Pad or crop data.

        Parameters
        ----------
        x
            original data
        dim
            dim along which padding should be applied
        padded_shape
            shape of padded data

        Returns
        -------
            data with shape padded_shape
        """
        # Adapt image size
        if padded_shape is not None:
            s = list(x.shape)
            for idx, idim in enumerate(dim):
                s[idim] = padded_shape[idx]
            x = change_data_shape(x, tuple(s))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pad or crop data.

        Parameters
        ----------
        x
            original data with shape orig_shape

        Returns
        -------
            data with shape padded_shape
        """
        return self._pad_data(x, self.dim, self.padded_shape)

    def adjoint(self, x: torch.Tensor) -> torch.Tensor:
        """Crop or pad data.

        Parameters
        ----------
        x
            original data with shape padded_shape

        Returns
        -------
            data with shape orig_shape
        """
        return self._pad_data(x, self.dim, self.orig_shape)
