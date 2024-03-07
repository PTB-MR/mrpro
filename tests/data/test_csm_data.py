"""Tests the CsmData class."""

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

import dataclasses

import pytest
import torch
from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.phantoms.coils import birdcage_2d

from tests.helper import relative_image_difference


def multi_coil_image(n_coils, ph_ellipse, random_kheader):
    """Create multi-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ph_ellipse.n_y, x=ph_ellipse.n_x)

    # Create reference coil sensitivities
    csm_ref = birdcage_2d(n_coils, image_dimensions)

    # Create multi-coil phantom image data
    img = ph_ellipse.phantom.image_space(image_dimensions)
    # +1 to ensure that there is signal everywhere, for voxel == 0 csm cannot be determined.
    img_multi_coil = (img + 1) * csm_ref
    idata = IData.from_tensor_and_kheader(data=img_multi_coil, kheader=random_kheader)
    return (idata, csm_ref)


def test_CsmData_is_frozen_dataclass(random_test_data, random_kheader):
    """CsmData inherits frozen dataclass property from QData."""
    csm = CsmData(data=random_test_data, header=random_kheader)
    with pytest.raises(dataclasses.FrozenInstanceError):
        csm.data = random_test_data


def test_CsmData_iterative_Walsh(ellipse_phantom, random_kheader):
    """CsmData obtained with the iterative Walsh method."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = CsmData.from_idata_walsh(idata, smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm.data), torch.abs(csm_ref)) <= 0.01


@pytest.mark.cuda()
def test_CsmData_iterative_Walsh_cuda(ellipse_phantom, random_kheader):
    """CsmData obtained with the iterative Walsh method in CUDA memory."""
    idata, csm_ref = multi_coil_image(n_coils=4, ph_ellipse=ellipse_phantom, random_kheader=random_kheader)

    # Estimate coil sensitivity maps
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = CsmData.from_idata_walsh(idata.cuda(), smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert relative_image_difference(torch.abs(csm.data), torch.abs(csm_ref.cuda())) <= 0.01
