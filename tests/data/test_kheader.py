"""Tests for KHeader class."""

import pytest
import torch
from mrpro.data import KHeader
from mrpro.data.traj_calculators.KTrajectoryCalculator import DummyTrajectory


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
    assert kheader.fa is not None
    assert torch.allclose(kheader.fa, fa_default)
    # tr is not mandatory but overwritten with None
    assert kheader.tr is tr_default
