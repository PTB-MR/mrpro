"""Test FastMRI dataset"""

import h5py
import pytest
from mrpro.data import KData
from mrpro.phantoms.fastmri import FastMRIDataset
from mrpro.utils import RandomGenerator

MINIMAL_HEADER_XML = r"""<?xml version="1.0" encoding="utf-8"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xs="http://www.w3.org/2001/XMLSchema"
xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd">
   <studyInformation>
      <studyTime>15:05:07</studyTime>
   </studyInformation>
   <measurementInformation>
      <measurementID>25077_449075862_449075873_819</measurementID>
      <patientPosition>HFS</patientPosition>
      <protocolName>AX</protocolName>
      <frameOfReferenceUID>1.3.12.2.1107.5.2.30.25077.1.20180401144405156.0.0.4998</frameOfReferenceUID>
   </measurementInformation>
   <acquisitionSystemInformation>
      <systemVendor>SIEMENS</systemVendor>
      <systemModel>Avanto</systemModel>
      <systemFieldStrength_T>1.494</systemFieldStrength_T>
      <relativeReceiverNoiseBandwidth>0.793</relativeReceiverNoiseBandwidth>
      <receiverChannels>4</receiverChannels>
      <institutionName>NYU</institutionName>
   </acquisitionSystemInformation>
   <experimentalConditions>
      <H1resonanceFrequency_Hz>63646310</H1resonanceFrequency_Hz>
   </experimentalConditions>
   <encoding>
      <encodedSpace>
         <matrixSize>
            <x>640</x>
            <y>322</y>
            <z>1</z>
         </matrixSize>
         <fieldOfView_mm>
            <x>440</x>
            <y>221.98</y>
            <z>7.5</z>
         </fieldOfView_mm>
      </encodedSpace>
      <reconSpace>
         <matrixSize>
            <x>320</x>
            <y>320</y>
            <z>1</z>
         </matrixSize>
         <fieldOfView_mm>
            <x>220</x>
            <y>220</y>
            <z>5</z>
         </fieldOfView_mm>
      </reconSpace>
      <trajectory>cartesian</trajectory>
      <encodingLimits>
         <kspace_encoding_step_1>
            <minimum>0</minimum>
            <maximum>225</maximum>
            <center>113</center>
         </kspace_encoding_step_1>
         <kspace_encoding_step_2>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </kspace_encoding_step_2>
         <average>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </average>
         <slice>
            <minimum>0</minimum>
            <maximum>33</maximum>
            <center>0</center>
         </slice>
         <contrast>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </contrast>
         <phase>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </phase>
         <repetition>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </repetition>
         <set>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </set>
         <segment>
            <minimum>0</minimum>
            <maximum>0</maximum>
            <center>0</center>
         </segment>
      </encodingLimits>
      <parallelImaging>
         <accelerationFactor>
            <kspace_encoding_step_1>1</kspace_encoding_step_1>
            <kspace_encoding_step_2>1</kspace_encoding_step_2>
         </accelerationFactor>
         <calibrationMode>other</calibrationMode>
         <interleavingDimension>other</interleavingDimension>
      </parallelImaging>
   </encoding>
   <sequenceParameters>
      <TR>500</TR>
      <TE>9.3</TE>
      <TI>100</TI>
      <flipAngle_deg>140</flipAngle_deg>
      <sequence_type>TurboSpinEcho</sequence_type>
      <echo_spacing>9.3</echo_spacing>
   </sequenceParameters>
</ismrmrdHeader>
"""

N_COILS = 2
N_K0 = 640
N_K1 = 322
N_SLICES_BRAIN = 2
N_SLICES_KNEE = 3


@pytest.fixture(scope='session')
def mock_fastmri_data(tmp_path_factory):
    """Create temporary HDF5 files mimicking FastMRI data structure."""

    test_dir = tmp_path_factory.mktemp('fastmri_test_data')
    rng = RandomGenerator(0)

    with h5py.File(test_dir / 'brain_file.h5', 'w') as f:
        kspace_brain = rng.complex64_tensor((N_COILS, N_SLICES_BRAIN, N_K0, N_K1)).numpy()
        f.create_dataset('kspace', data=kspace_brain)
        f.attrs['acquisition'] = 'AXT2_FLAIR'
        f.create_dataset('ismrmrd_header', data=MINIMAL_HEADER_XML.encode('utf-8'))

    with h5py.File(test_dir / 'knee_file.h5', 'w') as f:
        kspace_knee = rng.complex64_tensor((N_SLICES_KNEE, N_COILS, N_K0, N_K1)).numpy()
        f.create_dataset('kspace', data=kspace_knee)
        f.attrs['acquisition'] = 'CORPDFS_FBK'
        f.create_dataset('ismrmrd_header', data=MINIMAL_HEADER_XML.encode('utf-8'))

    return test_dir


def test_fastmri_dataset_init_len(mock_fastmri_data):
    """Test dataset initialization and length calculation."""
    dataset = FastMRIDataset(data_path=mock_fastmri_data)
    assert len(dataset) == N_SLICES_BRAIN + N_SLICES_KNEE


def test_fastmri_dataset_getitem(mock_fastmri_data):
    """Test __getitem__ for both brain and knee slices."""
    dataset = FastMRIDataset(data_path=mock_fastmri_data)

    # Test brain slice (index within the first file)
    idx_brain = N_SLICES_BRAIN // 2
    kdata_brain = dataset[idx_brain]
    assert isinstance(kdata_brain, KData)
    assert kdata_brain.shape == (1, N_COILS, 1, N_K1, N_K0)

    # Test knee slice (index within the second file)
    idx_knee = N_SLICES_BRAIN + N_SLICES_KNEE // 2
    kdata_knee = dataset[idx_knee]
    assert isinstance(kdata_knee, KData)
    assert kdata_knee.shape == (1, N_COILS, 1, N_K1, N_K0)

    # Test negative index
    kdata_last = dataset[-1]
    assert kdata_last.data.shape == (1, N_COILS, 1, N_K1, N_K0)


def test_fastmri_dataset_getitem_out_of_bounds(mock_fastmri_data):
    """Test that accessing invalid indices raises IndexError."""
    dataset = FastMRIDataset(data_path=mock_fastmri_data)
    with pytest.raises(IndexError):
        _ = dataset[100]
    with pytest.raises(IndexError):
        _ = dataset[-100]
