from __future__ import annotations

from collections.abc import Sequence

import torch
from ptwt.conv_transform import wavedec
from ptwt.conv_transform_2 import wavedec2
from ptwt.conv_transform_3 import wavedec3

from mrpro.operators import LinearOperator


class WaveletOp(LinearOperator):
    """Wavelet operator class."""

    def __init__(
        self,
        domain_shape: Sequence[int] | None = None,
        dim: Sequence[int] = (-3, -2, -1),
        wave_name: str = 'db4',
        level: int = 2,
    ):
        super().__init__()
        self.domain_shape = domain_shape
        self.wave_name = wave_name
        self.dim = dim
        self.level = level

        if domain_shape is not None:
            # calculate "coeff_slices"
            pass

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        if len(self.dim) == 1:
            coeff_list = wavedec(x, self.wave_name, level=self.level)
            coeff_list = self._format_coeffs_1d(coeff_list)
        elif len(self.dim) == 2:
            coeff_list = wavedec2(x, self.wave_name, level=self.level)
            coeff_list = self._format_coeffs_2d(coeff_list)
        elif len(self.dim) == 3:
            coeff_list = wavedec3(x, self.wave_name, level=self.level)
            coeff_list = self._format_coeffs_3d(coeff_list)

    def adjoint(self, x: tuple[torch.Tensor,]) -> tuple[torch.Tensor,]:
        return x

    def _format_coeffs_1d(self, coeffs: list[torch.Tensor]) -> list[torch.Tensor | tuple[torch.Tensor,]]:
        """Format 1D wavelet coefficients to MRpro format.

        Converts from   [a, d_n, ..., d_1]
        to              [a, (d_n,), ..., (d_1,)]
        """
        return [coeffs[0]] + [(c,) for c in coeffs[1:]]

    def _format_coeffs_2d(
        self,
        coeffs: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> list[torch.Tensor | tuple[torch.Tensor,]]:
        """Format 2D wavelet coefficients to MRpro format.

        At the moment, this function just returns the input coeffs as is.
        """
        return coeffs

    def _format_coeffs_3d(
        self, coeffs: list[torch.Tensor | dict[str, torch.Tensor]]
    ) -> list[torch.Tensor | tuple[torch.Tensor]]:
        """Format 3D wavelet coefficients to MRpro format.

        Converts from   [aaa, {aad_n, ada_n, add_n, ...}, ..., {aad_1, ada_1, add_1, ...}]
        to              [aaa, (aad_n, ada_n, add_n, ...), ..., (aad_1, ada_1, add_1, ...)]
        """
        res = [coeffs.pop(0)]
        for c_dict in coeffs:
            res.append(tuple(c_dict.values()))
        return res
