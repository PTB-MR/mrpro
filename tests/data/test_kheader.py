"""Tests for KHeader class."""

import torch
from ismrmrd.xsd.ismrmrdschema.ismrmrd import ismrmrdHeader
from mrpro.data import KHeader
from mrpro.data.AcqInfo import AcqInfo


def test_kheader_overwrite_parameter(random_mandatory_ismrmrd_header: ismrmrdHeader, random_acq_info: AcqInfo) -> None:
    """Overwrite existing parameter in KHeader."""
    overwrite = {'lamor_frequency_proton': 42}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, overwrite=overwrite)
    assert kheader.lamor_frequency_proton == 42


def test_kheader_set_missing_defaults(random_mandatory_ismrmrd_header: ismrmrdHeader, random_acq_info: AcqInfo) -> None:
    """Set missing value via defaults."""
    defaults = {'measurement_id': 42}
    kheader = KHeader.from_ismrmrd(random_mandatory_ismrmrd_header, random_acq_info, defaults=defaults)
    assert kheader.measurement_id == 42


def test_kheader_verify_None(random_mandatory_ismrmrd_header: ismrmrdHeader, random_acq_info: AcqInfo) -> None:
    """Correct handling of `None` and missing values in `KHeader` creation."""
    tr_default = None
    fa_default = torch.as_tensor([0.1])
    defaults = {'tr': tr_default, 'fa': fa_default}
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
    torch.testing.assert_close(torch.as_tensor(kheader_again.fa), kheader.fa)
