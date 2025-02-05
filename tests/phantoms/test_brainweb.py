import gzip
import re

import h5py
import pytest
from mrpro.phantoms.brainweb import CLASSES, OVERVIEW_URL, VERSION, download_brainweb

from tests import RandomGenerator


@pytest.fixture
def mock_requests(mocker):
    """Mock `requests.get` to return fake HTML for subjects and dummy gzipped data for class downloads."""

    def mock_get(url, timeout):
        if url == OVERVIEW_URL:
            return mocker.Mock(text='<option value=1><option value=2>')
        match = re.search(r'do_download_alias=subject(\d+)_(\w+)', url)
        if match:
            subject, class_name = match.groups()
            rng = RandomGenerator(int(subject) * 100 + int(class_name))
            fake_data = rng.int16_tensor((362, 434, 362), low=0, high=4096)
            compressed_data = gzip.compress(fake_data)
            return mocker.Mock(content=compressed_data)

    return mocker.patch('requests.get', side_effect=mock_get)


def test_download_brainweb(tmp_path, mock_requests):
    """Test download_brainweb"""

    download_brainweb(output_directory=tmp_path, workers=1, progress=False, compress=True)

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
