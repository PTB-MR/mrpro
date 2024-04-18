"""Tests for KHeader class."""

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

import datetime

import pytest
import torch
from ismrmrd import xsd
from mrpro.data import KHeader


def test_kheader_from_mandatory_ismrmrd_header(random_mandatory_ismrmrd_header, random_acq_info):
    KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info)


def test_kheader_fail_from_mandatory_ismrmrd_header(random_mandatory_ismrmrd_header, random_acq_info):
    with pytest.raises(ValueError, match='Could not create Header'):
        _ = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info)


def test_kheader_overwrite_missing_parameter(random_mandatory_ismrmrd_header, random_acq_info):
    overwrite = {
        'trajectory': xsd.trajectoryType('other'),
        'n_coils': 1,
        'datetime': datetime.datetime.now(),
        'te': torch.tensor([0.01]),
        'ti': torch.tensor([1.0]),
        'fa': torch.tensor([10.0]),
        'tr': torch.tensor([0.1]),
        'echo_spacing': torch.tensor([0.001]),
    }
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, overwrite=overwrite)
    assert kheader is not None


def test_kheader_set_missing_defaults(random_mandatory_ismrmrd_header, random_acq_info):
    defaults = {
        'trajectory': xsd.trajectoryType('other'),
        'n_coils': 1,
        'datetime': datetime.datetime.now(),
        'te': torch.tensor([1]),
        'ti': torch.tensor([1]),
        'fa': torch.tensor([1]),
        'tr': torch.tensor([1]),
        'echo_spacing': torch.tensor([1]),
    }
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, defaults=defaults)
    assert kheader is not None
