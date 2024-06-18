"""Class for coil sensitivity maps (csm)."""

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

from typing import TYPE_CHECKING

import torch

from mrpro.data.IData import IData
from mrpro.data.QData import QData
from mrpro.data.SpatialDimension import SpatialDimension

if TYPE_CHECKING:
    from mrpro.operators.SensitivityOp import SensitivityOp


class CsmData(QData):
    """Coil sensitivity map class."""

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        smoothing_width: int | SpatialDimension[int] = 5,
        power_iterations: int = 3,
        chunk_size_otherdim: int | None = None,
    ) -> CsmData:
        """Create csm object from image data using iterative Walsh method.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            width of smoothing filter.
        power_iterations
            number of iterations used to determine dominant eigenvector
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is None, which means that all elements are processed at once.
        """
        from mrpro.algorithms.csm.iterative_walsh import iterative_walsh

        # convert smoothing_width to SpatialDimension if int
        if isinstance(smoothing_width, int):
            smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)

        csm_fun = torch.vmap(
            lambda img: iterative_walsh(img, smoothing_width, power_iterations),
            chunk_size=chunk_size_otherdim,
        )
        csm_tensor = csm_fun(idata.data)
        csm = cls(header=idata.header, data=csm_tensor)
        return csm

    def as_operator(self) -> SensitivityOp:
        """Create SensitivityOp using a copy of the CSMs."""
        from mrpro.operators.SensitivityOp import SensitivityOp

        return SensitivityOp(self)
