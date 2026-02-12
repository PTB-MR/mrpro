"""Test MDCNN dataset."""

import io
import re
import urllib.request

import numpy as np
import pytest
import scipy.io
import torch
from mr2.data import KData
from mr2.phantoms.mdcnn import MDCNNDataset, download_mdcnn
from mr2.utils import RandomGenerator

N_PHASES = 2
N_COILS = 3
N_K1 = 208
N_K0 = 416
N_K0_PADDED = 823


@pytest.fixture(scope='session')
def mock_mdcnn_data(tmp_path_factory):
    """Create temporary numpy files mimicking MDCNN data."""
    test_dir = tmp_path_factory.mktemp('mdcnn_test_data')
    data = RandomGenerator(0).complex64_tensor((N_PHASES, N_COILS, 1, N_K1, N_K0))
    np.save(test_dir / 'P1.npy', data.numpy())
    return test_dir


@pytest.fixture
def mock_requests(monkeypatch):
    """Mock urllib.request.urlopen to return fake MDCNN data."""

    def create_mat_data(file_id):
        """Create fake .mat file data with the required structure."""
        data = RandomGenerator(file_id).float32_tensor((1, N_PHASES, N_K0_PADDED, N_K1, N_COILS, 2)).numpy()
        buffer = io.BytesIO()
        scipy.io.savemat(buffer, {'data': data})
        buffer.seek(0)
        return buffer.read()

    def mock_urlopen(request):
        """Mock URL open returning fake MDCNN data."""
        url = request.full_url
        if match := re.search(r'datafile/(\d+)', url):
            file_id = int(match.group(1))

            # Create mock response using simple type creation
            mock_resp = type(
                'MockResponse',
                (),
                {
                    'headers': {'Content-Disposition': f'attachment; filename=P{file_id}_data.mat'},
                    'read': lambda _: create_mat_data(file_id),
                    '__enter__': lambda self: self,
                    '__exit__': lambda *_: None,
                },
            )()
            return mock_resp

        raise ValueError(f'Unexpected URL: {url}')

    monkeypatch.setattr(urllib.request, 'urlopen', mock_urlopen)


def test_download_mdcnn(tmp_path, mock_requests):
    """Test download_mdcnn using mock data."""
    download_mdcnn(output_directory=tmp_path, n_files=2, workers=1, progress=False)

    # Check that numpy files were created
    npy_files = list(tmp_path.glob('P*.npy'))
    assert len(npy_files) == 2  # 2 subjects requested

    # Verify data shape and type
    data = np.load(npy_files[0])
    assert isinstance(data, np.ndarray)
    assert data.dtype == np.complex64
    assert data.shape == (N_PHASES, N_COILS, 1, N_K1, N_K0)


def test_mdcnn_dataset_getitem(mock_mdcnn_data):
    """Test dataset __getitem__ method."""
    dataset = MDCNNDataset(path=mock_mdcnn_data)
    sample = dataset[0]

    assert isinstance(sample, KData)
    assert sample.shape == (N_PHASES, N_COILS, 1, N_K1, N_K0)
    assert sample.data.dtype == torch.complex64
    assert not torch.isnan(sample.data).any()

    # Test header attributes
    assert sample.header.recon_matrix.x == 208
    assert sample.header.recon_matrix.y == 208
    assert sample.header.encoding_matrix.x == 624
    assert sample.header.encoding_matrix.y == 624


def test_mdcnn_dataset_init(mock_mdcnn_data):
    """Test MDCNNDataset initialization."""
    dataset = MDCNNDataset(path=mock_mdcnn_data)
    assert len(dataset) == 1  # One test file
