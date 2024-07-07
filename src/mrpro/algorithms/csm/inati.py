"""Inati method for coil sensitivity map calculation."""

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
from mrpro.utils.sliding_window import sliding_window


def inati(
    coil_images: torch.Tensor,
    kernel_size: SpatialDimension[int] | int,
    n_power_iterations: int,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using an the Inati method [1]_ [2]_.

    This is for a single set of coil images. The input should be a tensor with dimensions (coils, z, y, x). The output
    will have the same dimensions. Either apply this function individually to each set of coil images, or see
    CsmData.from_idata_walsh which performs this operation on a whole dataset.

    .. [1] S. Inati, M. Hansen, and P. Kellman, A solution to the phase problem in adaptvie coil combination.
       in Proceedings of the 21st Annual Meeting of ISMRM, Salt Lake City, USA, 2672 (2013).

    .. [2] S. Inati, and M. Hansen, A Fast Optimal Method for Coil Sensitivity Estimation and Adaptive Coil Combination
       for Complex Images. in Proceedings of Joint Annual Meeting ISMRM-ESMRMB, Milan, Italy, 7115 (2014).

    Parameters
    ----------
    coil_images
        images for each coil element
    kernel_size
        kernel size
    power_iterations
        number of iterations used to determine dominant eigenvector
    """
    padding_mode = 'circular'
    if isinstance(kernel_size, int):
        kernel_size = SpatialDimension(kernel_size, kernel_size, kernel_size)

    if any([ks % 2 != 1 for ks in [kernel_size.z, kernel_size.y, kernel_size.x]]):
        raise ValueError('kernel_size must be odd')
    if n_power_iterations < 1:
        raise ValueError('power must be at least 1')

    halfKs = [ks // 2 for ks in kernel_size.zyx]
    padded = torch.nn.functional.pad(
        coil_images, (halfKs[-3], halfKs[-3], halfKs[-2], halfKs[-2], halfKs[-1], halfKs[-1]), mode=padding_mode
    )
    D = sliding_window(padded, kernel_size.zyx, axis=(-3, -2, -1)).flatten(-2)  # coil E1, E0, ks*ks
    DH_D = torch.einsum('i...j,k...j->...ik', D, D.conj())  # E1,E0,coil,coil
    singular_vector = torch.sum(D, dim=-1)  # coil, E1, E0
    singular_vector /= singular_vector.abs().square().sum(0, keepdim=True).sqrt()
    for _ in range(n_power_iterations):
        singular_vector = torch.einsum('...ij,j...->i...', DH_D, singular_vector)  # coil, E1, E0
        singular_vector /= singular_vector.abs().square().sum(0, keepdim=True).sqrt()
    singular_value = torch.einsum('i...j,i...->...j', D, singular_vector)  # E1, E0, ks*ks
    phase = singular_value.sum(-1)
    phase /= phase.abs()  # E1, E0
    csm = singular_vector.conj() * phase[None, ...]
    return csm
