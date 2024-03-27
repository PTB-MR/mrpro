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

import torch

from mrpro.algorithms.csm._iterative_walsh import iterative_walsh
from mrpro.data._IData import IData
from mrpro.data._QData import QData
from mrpro.data._SpatialDimension import SpatialDimension
from mrpro.utils.filters import uniform_filter_3d


class CsmData(QData):
    """Coil sensitivity map class."""

    @staticmethod
    def _iterative_walsh_csm(
        coil_images: torch.Tensor,
        smoothing_width: SpatialDimension[int],
        power_iterations: int,
    ) -> torch.Tensor:
        """Calculate csm using an iterative version of the Walsh method.

        This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

        More information on the method can be found in
        https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

        Parameters
        ----------
        coil_images
            images for each coil element.
        smoothing_width
            width smoothing filter.
        power_iterations
            number of iterations used to determine dominant eigenvector
        """
        # Compute the pointwise covariance between coils
        coil_cov = torch.einsum('azyx,bzyx->abzyx', coil_images, coil_images.conj())

        # Smooth the covariance along y-x for 2D and z-y-x for 3D data
        coil_cov = uniform_filter_3d(coil_cov, filter_width=smoothing_width)

        # At each point in the image, find the dominant eigenvector
        # of the signal covariance matrix using the power method
        v = coil_cov.sum(dim=0)
        for _ in range(power_iterations):
            v /= v.norm(dim=0)
            v = torch.einsum('abzyx,bzyx->azyx', coil_cov, v)
        csm_data = v / v.norm(dim=0)

        # Make sure there are no inf or nan-values due to very small values in the covariance matrix
        # nan_to_num does not work for complexfloat, boolean indexing not with vmap.
        csm_data = torch.where(torch.isfinite(csm_data), csm_data, 0.0)
        return csm_data

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
        # convert smoothing_width to SpatialDimension if int
        if isinstance(smoothing_width, int):
            smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)

        csm_fun = torch.vmap(
            lambda img: iterative_walsh(img, smoothing_width, power_iterations),
            chunk_size=chunk_size_otherdim,
        )
        csm_data = csm_fun(idata.data)

        return cls(header=idata.header, data=csm_data)
