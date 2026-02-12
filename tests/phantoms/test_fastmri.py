"""Test FastMRI dataset"""

import h5py
import pytest
import torch
from mr2.data import KData
from mr2.phantoms import FastMRIImageDataset, FastMRIKDataDataset
from mr2.utils import RandomGenerator

N_COILS_BRAIN = 16
N_COILS_KNEE = 15
N_K0 = 640
N_K1_BRAIN = 322
N_K1_KNEE = 368
N_SLICES_BRAIN = 2
N_SLICES_KNEE = 3


@pytest.fixture(scope='session')
def mock_fastmri_brain_data(tmp_path_factory):
    """Create temporary HDF5 files mimicking FastMRI brain data."""
    test_dir = tmp_path_factory.mktemp('fastmri_brain_test_data')
    rng = RandomGenerator(0)
    header = r"""<?xml version="1.0" encoding="utf-8"?>
    <ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd">
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
    </encoding>
    </ismrmrdHeader>
"""
    with h5py.File(test_dir / 'brain_file.h5', 'w') as f:
        kspace_brain = rng.complex64_tensor((N_SLICES_BRAIN, N_COILS_BRAIN, N_K0, N_K1_BRAIN)).numpy()
        f.create_dataset('kspace', data=kspace_brain)
        f.attrs['acquisition'] = 'AXT2_FLAIR'
        f.create_dataset('ismrmrd_header', data=header.encode('utf-8'))

    return test_dir


@pytest.fixture(scope='session')
def mock_fastmri_knee_data(tmp_path_factory):
    """Create temporary HDF5 files mimicking FastMRI knee data."""
    test_dir = tmp_path_factory.mktemp('fastmri_knee_test_data')
    rng = RandomGenerator(1)
    header = r"""<?xml version="1.0" encoding="utf-8"?>
    <ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD" xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.ismrm.org/ISMRMRD ismrmrd.xsd">
        <experimentalConditions>
        <H1resonanceFrequency_Hz>63646310</H1resonanceFrequency_Hz>
    </experimentalConditions>
    <encoding>
        <encodedSpace>
            <matrixSize>
                <x>640</x>
                <y>368</y>
                <z>1</z>
            </matrixSize>
            <fieldOfView_mm>
                <x>280</x>
                <y>161</y>
                <z>4.5</z>
            </fieldOfView_mm>
        </encodedSpace>
        <reconSpace>
            <matrixSize>
                <x>320</x>
                <y>320</y>
                <z>1</z>
            </matrixSize>
            <fieldOfView_mm>
                <x>140</x>
                <y>140</y>
                <z>3</z>
            </fieldOfView_mm>
        </reconSpace>
    </encoding>
    </ismrmrdHeader>
    """
    with h5py.File(test_dir / 'knee_file.h5', 'w') as f:
        kspace_knee = rng.complex64_tensor((N_SLICES_KNEE, N_COILS_KNEE, N_K0, N_K1_KNEE)).numpy()
        f.create_dataset('kspace', data=kspace_knee)
        f.attrs['acquisition'] = 'CORPDFS_FBK'
        f.create_dataset('ismrmrd_header', data=header.encode('utf-8'))

    return test_dir


@pytest.mark.parametrize(
    ('data_fixture', 'n_slices', 'n_coils', 'n_k1'),
    [
        ('mock_fastmri_brain_data', N_SLICES_BRAIN, N_COILS_BRAIN, N_K1_BRAIN),
        ('mock_fastmri_knee_data', N_SLICES_KNEE, N_COILS_KNEE, N_K1_KNEE),
    ],
)
def test_fastmri_kdata_dataset_single_slice(request, data_fixture, n_slices, n_coils, n_k1):
    """Test KDataDataset for both brain and knee data."""
    data_path = request.getfixturevalue(data_fixture)
    dataset = FastMRIKDataDataset(path=data_path, single_slice=True)
    assert len(dataset) == n_slices

    kdata = dataset[n_slices // 2]
    assert isinstance(kdata, KData)
    assert kdata.shape == (1, n_coils, 1, n_k1, N_K0)

    kdata_last = dataset[-1]
    assert kdata_last.data.shape == (1, n_coils, 1, n_k1, N_K0)


@pytest.mark.parametrize(
    ('data_fixture', 'n_slices', 'n_coils', 'n_k1'),
    [
        ('mock_fastmri_brain_data', N_SLICES_BRAIN, N_COILS_BRAIN, N_K1_BRAIN),
        ('mock_fastmri_knee_data', N_SLICES_KNEE, N_COILS_KNEE, N_K1_KNEE),
    ],
)
def test_fastmri_kdata_dataset_stack_of_slices(request, data_fixture, n_slices, n_coils, n_k1):
    """Test KDataDataset for both brain and knee data."""
    data_path = request.getfixturevalue(data_fixture)
    dataset = FastMRIKDataDataset(path=data_path, single_slice=False)
    assert len(dataset) == 1

    kdata = dataset[0]
    assert isinstance(kdata, KData)
    assert kdata.shape == (n_slices, n_coils, 1, n_k1, N_K0)


@pytest.mark.parametrize(
    ('data_fixture', 'n_slices', 'n_coils'),
    [
        ('mock_fastmri_brain_data', N_SLICES_BRAIN, N_COILS_BRAIN),
        ('mock_fastmri_knee_data', N_SLICES_KNEE, N_COILS_KNEE),
    ],
)
@pytest.mark.parametrize('coil_combine', [False, True])
def test_fastmri_image_dataset(request, data_fixture, n_slices: int, n_coils: int, coil_combine: bool):
    """Test ImageDataset for both brain and knee data, with and without coil combination."""
    data_path = request.getfixturevalue(data_fixture)
    dataset = FastMRIImageDataset(path=data_path, coil_combine=coil_combine)
    expected_coils = 1 if coil_combine else n_coils
    assert len(dataset) == n_slices

    img = dataset[n_slices // 2]
    assert img.shape == (1, expected_coils, 1, 320, 320)
    assert img.dtype == torch.complex64
    assert not torch.isnan(img).any()


@pytest.mark.parametrize(
    ('data_fixture', 'n_coils'),
    [
        ('mock_fastmri_brain_data', N_COILS_BRAIN),
        ('mock_fastmri_knee_data', N_COILS_KNEE),
    ],
)
def test_fastmri_image_dataset_augment(request, data_fixture, n_coils: int):
    """Test ImageDataset augmentation for both brain and knee data."""

    def mock_augment(x, idx):
        return x * 2.0  # Simple multiplication augmentation

    data_path = request.getfixturevalue(data_fixture)
    dataset = FastMRIImageDataset(
        path=data_path,
        coil_combine=False,  # Test augmentation with multi-coil data
        augment=mock_augment,
    )

    img = dataset[0]
    assert img.shape == (1, n_coils, 1, 320, 320)
    img_no_aug = FastMRIImageDataset(path=data_path, coil_combine=False)[0]
    assert torch.allclose(img, img_no_aug * 2.0)


def test_fastmri_dataset_getitem_out_of_bounds(mock_fastmri_brain_data):
    """Test that accessing invalid indices raises IndexError."""
    dataset = FastMRIImageDataset(path=mock_fastmri_brain_data)
    with pytest.raises(IndexError):
        _ = dataset[100]
    with pytest.raises(IndexError):
        _ = dataset[-100]
