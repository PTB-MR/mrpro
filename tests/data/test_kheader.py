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

import pytest
import torch
from mrpro.data import KHeader
from mrpro.data.traj_calculators._KTrajectoryCalculator import DummyTrajectory


def test_kheader_fail_from_mandatory_ismrmrd_header(random_mandatory_ismrmrd_header, random_acq_info):
    """KHeader cannot be created only from ismrmrd header because trajectory is missing."""
    with pytest.raises(ValueError, match='Could not create Header'):
        _ = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info)


def test_kheader_overwrite_missing_parameter(random_mandatory_ismrmrd_header, random_acq_info):
    """KHeader can be created if trajectory is provided."""
    overwrite = {'trajectory': DummyTrajectory()}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, overwrite=overwrite)
    assert kheader is not None


def test_kheader_set_missing_defaults(random_mandatory_ismrmrd_header, random_acq_info):
    """KHeader can be created if default trajectory is defined."""
    defaults = {'trajectory': DummyTrajectory()}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, defaults=defaults)
    assert kheader is not None


def test_kheader_verify_None(random_mandatory_ismrmrd_header, random_acq_info):
    """Correct handling of None and missing values in KHeader creation."""
    tr_default = None
    fa_default = torch.as_tensor([0.1])
    defaults = {'trajectory': DummyTrajectory(), 'tr': tr_default, 'fa': fa_default}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, defaults=defaults)
    # ti is not mandatory
    assert kheader.ti is None
    # fa is not mandatory but overwriting with value
    assert kheader.fa is not None and torch.allclose(kheader.fa, fa_default)
    # tr is not mandatory but overwritten with None
    assert kheader.tr is tr_default
