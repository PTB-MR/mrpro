"""Tests the IData class."""

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

import numpy as np
import pytest

from mrpro.data import IData
from tests.data import Dicom2DTestImage
from tests.data.conftest import random_kheader
from tests.data.conftest import random_test_data
from tests.phantoms.test_phantoms import ph_ellipse


@pytest.fixture(scope='session')
def dcm_2d(ph_ellipse, tmp_path_factory):
    """Single 2D dicom image."""
    dcm_filename = tmp_path_factory.mktemp('mrpro') / 'dicom_2d.h5'
    dcm_idat = Dicom2DTestImage(filename=dcm_filename, phantom=ph_ellipse.phantom)
    return dcm_idat


def test_IData_from_dcm_file(dcm_2d):
    """IData from single dicom file."""
    idat = IData.from_single_dicom(dcm_2d.filename)
    np.testing.assert_almost_equal(np.abs(idat.data[0, 0, 0, ...]), np.moveaxis(dcm_2d.imref, (0, 1), (1, 0)))


def test_IData_from_kheader_and_tensor(random_kheader, random_test_data):
    """IData from KHeader and data tensor."""
    idat = IData.from_tensor_and_kheader(data=random_test_data, kheader=random_kheader)
    assert idat.header.te == random_kheader.te
