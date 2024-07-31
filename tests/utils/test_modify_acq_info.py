"""Tests for modification of acquisition infos."""

from einops import rearrange
from mrpro.utils import modify_acq_info


def test_modify_acq_info(random_kheader_shape):
    """Test the modification of the acquisition info."""
    # Create random header where AcqInfo fields are of shape [n_k1*n_k2] and reshape to [n_other, n_k2, n_k1]
    kheader, n_other, _, n_k2, n_k1, _ = random_kheader_shape

    def reshape_acq_data(data):
        return rearrange(data, '(other k2 k1) ... -> other k2 k1 ...', other=n_other, k2=n_k2, k1=n_k1)

    kheader.acq_info = modify_acq_info(reshape_acq_data, kheader.acq_info)

    # Verify shape
    assert kheader.acq_info.center_sample.shape == (n_other, n_k2, n_k1, 1)
