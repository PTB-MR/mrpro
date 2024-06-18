"""Iterative Walsh method for coil sensitivity map calculation."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

import torch

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.filters import uniform_filter


def iterative_walsh(
    coil_images: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
    power_iterations: int,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using an iterative version of the Walsh method.

    This is for a single set of coil images. The input should be a tensor with dimensions
    (coils, z, y, x). The output will have the same dimensions.
    Either apply this function individually to each set of coil images,
    or see CsmData.from_idata_walsh which performs this operation on a whole dataset.

    This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

    More information on the method can be found in
    https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

    Parameters
    ----------
    coil_images
        images for each coil element
    smoothing_width
        width of the smoothing filter
    power_iterations
        number of iterations used to determine dominant eigenvector
    """
    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)
    # Compute the pointwise covariance between coils
    coil_covariance = torch.einsum('azyx,bzyx->abzyx', coil_images, coil_images.conj())

    # Smooth the covariance along y-x for 2D and z-y-x for 3D data
    coil_covariance = uniform_filter(coil_covariance, width=smoothing_width.zyx, axis=(-3, -2, -1))

    # At each point in the image, find the dominant eigenvector
    # of the signal covariance matrix using the power method
    v = coil_covariance.sum(dim=0)
    for _ in range(power_iterations):
        v /= v.norm(dim=0)
        v = torch.einsum('abzyx,bzyx->azyx', coil_covariance, v)
    csm = v / v.norm(dim=0)

    # Make sure there are no inf or nan-values due to very small values in the covariance matrix
    # nan_to_num does not work for complexfloat, boolean indexing not with vmap.
    csm = torch.where(torch.isfinite(csm), csm, 0.0)
    return csm
