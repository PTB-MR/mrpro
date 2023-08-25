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

import pytest
from IsmrmrdRawData import IsmrmrdRawData

from mrpro.data import KData
from mrpro.data._KTrajectory import DummyTrajectory


@pytest.fixture(scope='session')
def ismrmrd_cart(tmp_path_factory):
    # Fully sampled cartesian data set
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawData(filename=ismrmrd_filename)
    ismrmrd_kdat.create()
    return (ismrmrd_kdat)


@pytest.fixture(scope='session')
def ismrmrd_cart_us4(tmp_path_factory):
    # Undersampled cartesian data set
    ismrmrd_filename = tmp_path_factory.mktemp('mrpro') / 'ismrmrd_cart.h5'
    ismrmrd_kdat = IsmrmrdRawData(filename=ismrmrd_filename, acceleration=4)
    ismrmrd_kdat.create()
    return (ismrmrd_kdat)


def test_KData_from_file(ismrmrd_cart):
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory())
    assert k is not None

# Expected to fail with DummyTrajectory


def test_KData_from_file_undersampled(ismrmrd_cart_us4):
    with pytest.raises(ValueError):
        KData.from_file(ismrmrd_cart_us4.filename, DummyTrajectory())


@pytest.mark.parametrize('field,value', [('b0', 11.3), ('tr', [24.3,])])
def test_KData_modify_header(ismrmrd_cart, field, value):
    # Overwrite some parameters
    par_dict = {field: value}
    k = KData.from_file(ismrmrd_cart.filename, DummyTrajectory(), header_overwrites=par_dict)
    assert getattr(k.header, field) == value
