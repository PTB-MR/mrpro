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

import generate_shepp_logan_dataset
import pytest

from mrpro.data import KData
from mrpro.data._KTrajectory import DummyTrajectory


def test_KData_from_file(tmp_path):
    # Create an example ismrmrd data set
    ismrmrd_filename = tmp_path / 'ismrmrd.h5'
    generate_shepp_logan_dataset.create(filename=ismrmrd_filename)

    k = KData.from_file(ismrmrd_filename, DummyTrajectory())
    assert k is not None


@pytest.mark.parametrize('field,value', [('b0', 11.3), ('tr', [24.3,])])
def test_KData_modify_header(tmp_path, field, value):
    # Create an example ismrmrd data set
    ismrmrd_filename = tmp_path / 'ismrmrd.h5'
    generate_shepp_logan_dataset.create(filename=ismrmrd_filename)

    # Overwrite some parameters
    par_dict = {field: value}
    k = KData.from_file(ismrmrd_filename, DummyTrajectory(), header_overwrites=par_dict)
    assert getattr(k.header, field) == value
