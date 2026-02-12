"""Test M4Raw dataset"""

import h5py
import pytest
from mr2.data import KData
from mr2.phantoms.m4raw import M4RawDataset
from mr2.utils import RandomGenerator

N_COILS = 2
N_K0 = 256
N_K1 = 195
N_K1_PADDED = 256
N_SLICES = 18
N_REPS = 2


@pytest.fixture(scope='session')
def m4raw_data(tmp_path_factory):
    """Create temporary HDF5 files mimicking M4Raw data."""
    test_dir = tmp_path_factory.mktemp('m4raw_test_data')
    rng = RandomGenerator(0)

    # xml header similar to what's in the M4Raw dataset, but
    # only including elements that are actually used in the M4RawDataset
    header = """<?xml version="1.0" encoding="utf-8"?>
    <ns0:ismrmrdHeader xmlns:ns0="http://www.ismrm.org/ISMRMRD" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <ns0:measurementInformation>
            <ns0:measurementID>Test_ID</ns0:measurementID>
            <ns0:protocolName>T1_SE_5mm_TRS</ns0:protocolName>
        </ns0:measurementInformation>
        <ns0:acquisitionSystemInformation>
            <ns0:systemVendor>XinGaoYi Medical Equipment Co.,Ltd.</ns0:systemVendor>
            <ns0:systemModel>XGY OPER-0.3</ns0:systemModel>
        </ns0:acquisitionSystemInformation>
        <ns0:encoding>
            <ns0:echoTrainLength>1</ns0:echoTrainLength>
        </ns0:encoding>
        <ns0:sequenceParameters>
            <ns0:TR>500.0</ns0:TR>
            <ns0:TE>18.4</ns0:TE>
            <ns0:TI>0.0</ns0:TI>
            <ns0:flipAngle_deg>180.0</ns0:flipAngle_deg>
            <ns0:sequence_type>SpinEcho_TRA</ns0:sequence_type>
            <ns0:echo_spacing>N/A</ns0:echo_spacing>
        </ns0:sequenceParameters>
    </ns0:ismrmrdHeader>
    """
    base_name = '1_T1'
    for rep in range(N_REPS):
        filename = test_dir / f'{base_name}{rep + 1}.h5'
        with h5py.File(filename, 'w') as f:
            kspace = rng.complex64_tensor((N_SLICES, N_COILS, N_K0, N_K1_PADDED)).numpy()
            f.create_dataset('kspace', data=kspace)
            f.create_dataset('ismrmrd_header', data=header.encode('utf-8'))

    return test_dir


def test_m4raw_dataset_single_slice(m4raw_data):
    """Test M4RawDataset with single_slice=True."""
    dataset = M4RawDataset(path=m4raw_data, single_slice=True)
    assert len(dataset) == N_SLICES
    kdata = dataset[N_SLICES // 2]
    assert isinstance(kdata, KData)
    assert kdata.shape == (N_REPS, N_COILS, 1, N_K1, N_K0)


def test_m4raw_dataset_stack_of_slices(m4raw_data):
    """Test M4RawDataset with single_slice=False."""
    dataset = M4RawDataset(path=m4raw_data, single_slice=False)
    assert len(dataset) == 1
    kdata = dataset[0]
    assert isinstance(kdata, KData)
    assert kdata.shape == (N_SLICES, N_REPS, N_COILS, 1, N_K1, N_K0)
