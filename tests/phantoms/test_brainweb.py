import gzip
import re
from typing import Literal

import h5py
import numpy as np
import pytest
import requests
import torch
from mrpro.phantoms.brainweb import (
    CLASSES,
    OVERVIEW_URL,
    VERSION,
    BrainwebSlices,
    BrainwebTissue,
    BrainwebVolumes,
    augment,
    download_brainweb,
    resize,
    trim_indices,
)

from tests import RandomGenerator


@pytest.fixture
def mock_requests(monkeypatch) -> None:
    """Mock `requests.get` return fake subject list and gzipped data."""

    def mock_get(url, timeout):
        """Mock HTTP GET requests dynamically extracting subject and class from the URL."""
        if url == OVERVIEW_URL:
            # Fake subject list
            return type('MockResponse', (), {'text': '<option value=1><option value=2>'})()

        if match := re.search(r'do_download_alias=subject(\d+)_([^&]*)', url):
            # Fake data
            subject, class_name = match.groups()
            rng = RandomGenerator(int(subject) * 100 + CLASSES.index(class_name))
            fake_data = rng.int16_tensor((362, 434, 362), low=0, high=4096)
            compressed_data = gzip.compress(fake_data.numpy().astype(np.uint16).tobytes(), compresslevel=0)
            return type('MockResponse', (), {'content': compressed_data, 'raise_for_status': lambda _: None})()

        raise ValueError(f'Unexpected URL: {url}')  # Ensure no unexpected requests are made

    monkeypatch.setattr(requests, 'get', mock_get)


def test_download_brainweb(tmp_path, mock_requests) -> None:
    """Test download_brainweb using mock data."""

    download_brainweb(output_directory=tmp_path, workers=1, progress=False, compress=False)

    # Check that HDF5 files were created
    hdf5_files = list(tmp_path.glob('s*.h5'))
    assert len(hdf5_files) == 2  # 2 subjects in mock data

    for hdf5_file in hdf5_files:
        with h5py.File(hdf5_file, 'r') as f:
            assert 'classes' in f
            assert 'classnames' in f.attrs
            assert 'subject' in f.attrs
            assert 'version' in f.attrs
            assert f.attrs['version'] == VERSION
            assert f['classes'].shape == (362, 434, 362, len(CLASSES) - 1)


@pytest.fixture(scope='session')
def brainweb_test_data(tmp_path_factory):
    """Create temporary HDF5 files for BrainwebVolumes testing."""
    test_dir = tmp_path_factory.mktemp('brainweb')

    for subject in range(2):  # Create test files for two subjects
        file_path = test_dir / f's{subject:02d}.h5'
        with h5py.File(file_path, 'w') as f:
            rng = RandomGenerator(int(subject))
            fake_data = rng.float32_tensor((362, 434, 362, len(CLASSES) - 1))
            fake_data *= 255 / fake_data.sum(-1, keepdim=True)
            f.create_dataset('classes', data=fake_data.to(torch.uint8))

            # Store metadata
            f.attrs['classnames'] = [c for c in CLASSES if c != 'bck']  # noqa: typos
            f.attrs['subject'] = subject
            f.attrs['version'] = 1
    return test_dir


def test_brainwebvolumes_init(brainweb_test_data) -> None:
    """Test BrainwebVolumes dataset initialization."""
    dataset = BrainwebVolumes(folder=brainweb_test_data)
    assert len(dataset) == 2  # Two subjects in test data


def test_brainweb_getitem(brainweb_test_data) -> None:
    """Test dataset __getitem__ method."""
    dataset = BrainwebVolumes(folder=brainweb_test_data, what=('m0', 'r1', 't2', 'mask', 'tissueclass', 'dura'))
    sample = dataset[0]

    assert isinstance(sample, dict)

    assert sample['m0'].shape == (362, 434, 362)
    assert sample['r1'].shape == (362, 434, 362)
    assert sample['t2'].shape == (362, 434, 362)
    assert sample['mask'].shape == (362, 434, 362)
    assert sample['tissueclass'].shape == (362, 434, 362)
    assert sample['dura'].shape == (362, 434, 362)

    assert sample['m0'].dtype == torch.complex64
    assert sample['r1'].dtype == torch.float
    assert sample['t2'].dtype == torch.float
    assert sample['mask'].dtype == torch.bool
    assert sample['tissueclass'].dtype == torch.long

    assert not torch.isnan(sample['m0']).any()
    assert not torch.isnan(sample['r1']).any()
    assert not torch.isnan(sample['dura']).any()


def test_brainweb_no_files(tmp_path) -> None:
    """Ensure error is raised if no files are found."""
    with pytest.raises(FileNotFoundError):
        BrainwebVolumes(folder=tmp_path)


def test_brainwebtissue_random_values_tuple() -> None:
    """Test BrainwebTissue when initialized with tuple parameters."""
    rng = torch.Generator().manual_seed(42)
    param = BrainwebTissue((0.5, 1.5), (0.02, 0.1), (0.7, 1.2), (-0.1, 0.1))

    assert isinstance(param.t1, tuple)
    assert isinstance(param.t2, tuple)
    assert isinstance(param.m0_abs, tuple)
    assert isinstance(param.m0_phase, tuple)

    assert param.t1[0] <= 1 / param.sample_r1(rng) <= param.t1[1]
    assert param.t2[0] <= 1 / param.sample_r2(rng) <= param.t2[1]

    m0 = param.sample_m0(rng)
    assert param.m0_abs[0] <= m0.abs() <= param.m0_abs[1]
    assert param.m0_phase[0] <= m0.angle() <= param.m0_phase[1]


def test_brainwebtissue_random_values_float() -> None:
    """Test BrainwebTissue when initialized with float parameters."""
    rng = torch.Generator().manual_seed(42)
    param = BrainwebTissue(1.0, 0.05, 0.9, 0.05)

    assert isinstance(param.t1, float)
    assert isinstance(param.t2, float)
    assert isinstance(param.m0_abs, float)
    assert isinstance(param.m0_phase, float)

    assert torch.tensor(param.t1).isclose(1 / param.sample_r1(rng))
    assert torch.tensor(param.t2).isclose(1 / param.sample_r2(rng))

    m0 = param.sample_m0(rng)
    assert torch.tensor(param.m0_abs).isclose(m0.abs())
    assert torch.tensor(param.m0_phase).isclose(m0.angle())


@pytest.mark.parametrize('size', [128, 256])
def test_affine_augment(size) -> None:
    """Test augment function."""
    data = RandomGenerator(1).float32_tensor((1, 150, 100))

    aug_data = augment(data, size, rng=torch.Generator().manual_seed(42))
    assert aug_data.shape == (1, size, size)
    assert aug_data.dtype == data.dtype

    aug_data_same = augment(data, size, rng=torch.Generator().manual_seed(42))
    torch.testing.assert_close(aug_data, aug_data_same)

    aug_data_different = augment(data, size, rng=torch.Generator().manual_seed(43))
    assert not torch.isclose(aug_data, aug_data_different).all()


@pytest.mark.parametrize('size', [128, 256])
def test_resize(size: int) -> None:
    """Text resize function."""
    data = RandomGenerator(2).float32_tensor((1, 150, 100))
    resized = resize(data, size)
    assert resized.shape == (1, size, size)
    assert resized.dtype == data.dtype


@pytest.mark.parametrize(
    ('mask', 'expected'),
    [
        (torch.tensor([[0, 0, 1], [0, 1, 1], [1, 1, 1]]), (slice(0, 3), slice(0, 3))),
        (torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), (slice(1, 2), slice(1, 2))),
    ],
)
def test_trim_indices(mask: torch.Tensor, expected: tuple[slice, slice]) -> None:
    """Test trim_indices function."""
    assert trim_indices(mask) == expected


def test_brainwebslices_init(brainweb_test_data) -> None:
    """Test BrainwebSlices dataset initialization."""
    dataset = BrainwebSlices(folder=brainweb_test_data, what=('m0', 'r1'), orientation='axial')
    assert len(dataset) > 0


def test_brainwebslices_getitem(brainweb_test_data) -> None:
    """Test dataset __getitem__ method."""
    dataset = BrainwebSlices(
        folder=brainweb_test_data, what=('m0', 'r1', 't2', 'mask', 'tissueclass', 'dura'), orientation='coronal'
    )
    sample = dataset[0]

    assert isinstance(sample, dict)

    assert sample['m0'].shape[-2:] == (256, 256)
    assert sample['r1'].shape[-2:] == (256, 256)
    assert sample['t2'].shape[-2:] == (256, 256)
    assert sample['mask'].shape[-2:] == (256, 256)
    assert sample['tissueclass'].shape[-2:] == (256, 256)
    assert sample['dura'].shape[-2:] == (256, 256)

    assert sample['m0'].dtype == torch.complex64
    assert sample['r1'].dtype == torch.float
    assert sample['t2'].dtype == torch.float
    assert sample['mask'].dtype == torch.bool
    assert sample['tissueclass'].dtype == torch.long

    assert not torch.isnan(sample['m0']).any()
    assert not torch.isnan(sample['r1']).any()


@pytest.mark.parametrize('orientation', ['axial', 'coronal', 'sagittal'])
def test_brainwebslices_orientations(brainweb_test_data, orientation: Literal['axial', 'coronal', 'sagittal']) -> None:
    """Test different slice orientations."""
    dataset = BrainwebSlices(folder=brainweb_test_data, orientation=orientation)
    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, dict)


def test_brainwebslices_no_files(tmp_path) -> None:
    """Ensure error is raised if no files are found."""
    with pytest.raises(FileNotFoundError):
        BrainwebSlices(folder=tmp_path)
