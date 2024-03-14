"""Class for Fast Fourier Operator."""

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

from collections.abc import Sequence

import torch

from mrpro.operators._LinearOperator import LinearOperator
from mrpro.operators._ZeroPadOp import ZeroPadOp


class FastFourierOp(LinearOperator):
    """Fast Fourier operator class."""

    def __init__(
        self,
        dim: Sequence[int] = (-3, -2, -1),
        recon_shape: Sequence[int] | None = None,
        encoding_shape: Sequence[int] | None = None,
    ) -> None:
        """Fast Fourier Operator class.

        Remark regarding the fftshift/ifftshift:
        fftshift shifts the zero-frequency point to the center of the data, ifftshift undoes this operation.
        The input to both forward and ajoint assumes that the zero-frequency is in the center of the data.
        Torch.fft.fftn and torch.fft.ifftn expect the zero-frequency to be the first entry in the tensor.
        Therefore for forward and ajoint first ifftshift needs to be applied, then fftn or ifftn and then ifftshift.

        Parameters
        ----------
        dim, optional
            dim along which FFT and IFFT are applied, by default last three dimensions (-1, -2, -3)
        encoding_shape, optional
            shape of encoded data
        recon_shape, optional
            shape of reconstructed data
        """
        super().__init__()
        self._dim = tuple(dim)
        self._pad_op: ZeroPadOp
        if encoding_shape is not None and recon_shape is not None:
            self._pad_op = ZeroPadOp(dim=dim, original_shape=recon_shape, padded_shape=encoding_shape)
        else:
            # No padding
            self._pad_op = ZeroPadOp(dim=(), original_shape=(), padded_shape=())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """FFT from image space to k-space.

        Parameters
        ----------
        x
            image data on Cartesian grid

        Returns
        -------
            FFT of x
        """
        y = torch.fft.fftshift(
            torch.fft.fftn(torch.fft.ifftshift(*self._pad_op.forward(x), dim=self._dim), dim=self._dim, norm='ortho'),
            dim=self._dim,
        )
        return (y,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """IFFT from k-space to image space.

        Parameters
        ----------
        y
            k-space data on Cartesian grid

        Returns
        -------
            IFFT of y
        """
        # FFT
        return self._pad_op.adjoint(
            torch.fft.fftshift(
                torch.fft.ifftn(torch.fft.ifftshift(y, dim=self._dim), dim=self._dim, norm='ortho'),
                dim=self._dim,
            ),
        )
