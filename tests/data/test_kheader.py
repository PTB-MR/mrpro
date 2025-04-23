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
