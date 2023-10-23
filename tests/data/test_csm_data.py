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

import numpy as np
import pytest

from mrpro.data import CsmData
from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.phantoms.coils import birdcage_2d
from tests.data.conftest import random_kheader
from tests.data.conftest import random_test_data
from tests.helper import rel_image_diff
from tests.phantoms.test_ellipse_phantom import ph_ellipse


def test_CsmData_is_frozen_dataclass(random_test_data, random_kheader):
    """CsmData inherits frozen dataclass property from QData."""
    csm = CsmData(data=random_test_data, header=random_kheader)
    with pytest.raises(dataclasses.FrozenInstanceError):
        csm.data = random_test_data


def test_CsmData_iterative_Walsh(ph_ellipse, random_kheader):
    """CsmData obtained with the iterative Walsh method."""
    num_coils = 4
    im_dim = SpatialDimension(z=1, y=ph_ellipse.ny, x=ph_ellipse.nx)

    # Create reference coil sensitivities
    csm_ref = birdcage_2d(num_coils, im_dim)

    # Create multi-coil phantom image data
    im = ph_ellipse.phantom.image_space(im_dim)
    # +1 to ensure that there is signal everywhere, for voxel == 0 csm cannot be determined.
    im_multi_coil = (im + 1) * csm_ref
    idat = IData.from_tensor_and_kheader(data=im_multi_coil, kheader=random_kheader)

    # Estimate coil sensitivity maps
    smoothing_width = SpatialDimension(z=1, y=5, x=5)
    csm = CsmData.from_idata_walsh(idat, smoothing_width)

    # Phase is only relative in csm calculation, therefore only the abs values are compared.
    assert rel_image_diff(np.abs(csm.data), np.abs(csm_ref)) <= 0.01
