"""Tests for KHeader class."""

import torch
from mrpro.data import KHeader
from mrpro.data.traj_calculators.KTrajectoryCalculator import DummyTrajectory


def test_kheader_overwrite_missing_parameter(random_mandatory_ismrmrd_header, random_acq_info) -> None:
    """KHeader can be created if trajectory is provided."""
    overwrite = {'trajectory': DummyTrajectory()}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, overwrite=overwrite)
    assert kheader is not None
    assert kheader.trajectory is overwrite['trajectory']


def test_kheader_set_missing_defaults(random_mandatory_ismrmrd_header, random_acq_info) -> None:
    """KHeader can be created if default trajectory is defined."""
    defaults = {'trajectory': DummyTrajectory()}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, defaults=defaults)
    assert kheader is not None
    assert kheader.trajectory is defaults['trajectory']


def test_kheader_verify_None(random_mandatory_ismrmrd_header, random_acq_info) -> None:
    """Correct handling of `None` and missing values in `KHeader` creation."""
    tr_default = None
    fa_default = torch.as_tensor([0.1])
    defaults = {'trajectory': DummyTrajectory(), 'tr': tr_default, 'fa': fa_default}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, defaults=defaults)
    # ti is not mandatory
    assert not kheader.ti
    # fa is not mandatory but overwriting with value
    assert isinstance(kheader.fa, torch.Tensor)
    assert torch.allclose(kheader.fa, fa_default)
    # tr is not mandatory but overwritten with None
    assert kheader.tr is tr_default


def test_kheader_to_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info):
    """Create ISMRMRD header from KHeader."""
    fa = [2.0, 3.0, 4.0, 5.0]
    overwrite = {'trajectory': DummyTrajectory(), 'fa': torch.deg2rad(torch.as_tensor(fa))}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, overwrite=overwrite)
    ismrmrd_header = kheader.to_ismrmrd()
    kheader_again = KHeader.from_ismrmrd(ismrmrd_header, random_acq_info, {'trajectory': DummyTrajectory()})
    assert ismrmrd_header.experimentalConditions.H1resonanceFrequency_Hz == kheader.lamor_frequency_proton
    assert ismrmrd_header.encoding[0].encodedSpace.matrixSize.z == kheader.encoding_matrix.zyx[0]
    assert ismrmrd_header.sequenceParameters.flipAngle_deg == fa
    torch.testing.assert_close(kheader_again.fa, kheader.fa)
