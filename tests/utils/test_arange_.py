import pytest
import torch
from mrpro.utils import arange_


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64], ids=['float32', 'int64'])
def test_arange_(dtype):
    """Test functionality of arange_."""
    tensor = torch.zeros(3, 4, dtype=dtype)
    arange_(tensor, dim=1)
    expected = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=tensor.dtype)
    torch.testing.assert_close(tensor, expected)


def test_arange_dim_out_of_range():
    """Test arange_ with a dimension out of range."""
    tensor = torch.zeros(3, 4)
    with pytest.raises(IndexError, match='Dimension 2 is out of range'):
        arange_(tensor, dim=2)
    with pytest.raises(IndexError, match='Dimension -3 is out of range'):
        arange_(tensor, dim=-3)
