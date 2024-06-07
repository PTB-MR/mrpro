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
        elif len(self.dim) == 2:
            coeff_list = wavedec2(x, self.wave_name, level=self.level)
        elif len(self.dim) == 3:
            coeff_list = wavedec3(x, self.wave_name, level=self.level)

        # convert coeff_list from ptwt to pywt dict format
        # convert coeff_dict to tenor using pywt similar function
        # return coeff tensor and nothing else

    def adjoint(self, x: tuple[torch.Tensor,]) -> tuple[torch.Tensor,]:
        return x

    def _coeff_to_array(self, coeff_list: list[torch.Tensor]) -> torch.Tensor:
        pass

    def _array_to_coeff(self, x: torch.Tensor) -> list[torch.Tensor]:
        pass
