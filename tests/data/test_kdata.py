"""Tests for the KData class."""

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
from IsmrmrdRawData import IsmrmrdRawData
from IsmrmrdRawData import k2i

from mrpro.data import KData
from mrpro.data._KTrajectory import DummyTrajectory


@pytest.fixture(scope='session')
def ismrmrd_cart(tmp_path_factory):
    # Fully sampled cartesian data set
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawData(filename=ismrmrd_filename, noise_level=0.0, repetitions=3)
    return ismrmrd_kdat


@pytest.fixture(scope='session')
def ismrmrd_cart_invalid_reps(tmp_path_factory):
    # Fully sampled cartesian data set
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawData(filename=ismrmrd_filename, noise_level=0.0, repetitions=3, flag_invalid_reps=True)
    return ismrmrd_kdat


@pytest.fixture(scope='session')
def ismrmrd_cart_us4(tmp_path_factory):
    # Undersampled cartesian data set
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawData(filename=ismrmrd_filename, acceleration=4)
    return ismrmrd_kdat


def test_KData_from_file(ismrmrd_cart):
    # Read in data from file
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert k is not None


def test_KData_kspace(ismrmrd_cart):
    # Read in data and verify k-space by comparing reconstructed image
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    irec = k2i(k.data, axes=(-1, -2))

    # Due to discretisation artefacts the reconstructed image will be different to the reference image. Using standard
    # testing functions such as numpy.testing.assert_almost_equal fails because there are few voxels with high
    # differences along the edges of the elliptic objects.
    idiff = np.mean(np.abs(irec[0, 0, 0, ...] - ismrmrd_cart.imref))
    imean = 0.5*np.mean(np.abs(irec[0, 0, 0, ...]) + np.abs(ismrmrd_cart.imref))
    assert idiff <= imean*0.05


def test_KData_from_file_undersampled(ismrmrd_cart_us4):
    # Expected to fail with DummyTrajectory
    with pytest.raises(ValueError):
        KData.from_file(ismrmrd_cart_us4.filename, DummyTrajectory())


# ToDo: This test is temporary - once Cartesian trajectories are properly calculated, this should be removed/adapted
def test_KData_from_file_diff_nky_for_rep(ismrmrd_cart_invalid_reps):
    # Multiple repetitions with different number of phase encoding lines
    with pytest.raises(ValueError, match='Number of k1 points in repetition: 128. Expected: 256'):
        KData.from_file(ismrmrd_cart_invalid_reps.filename, DummyTrajectory())


@pytest.mark.parametrize('field,value', [('b0', 11.3), ('tr', [24.3,])])
def test_KData_modify_header(ismrmrd_cart, field, value):
    # Overwrite some parameters in the header
    par_dict = {field: value}
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory(), header_overwrites=par_dict)
    assert getattr(k.header, field) == value
