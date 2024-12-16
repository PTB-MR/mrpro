import pytest
import torch
from mrpro.utils.indexing import Indexer


@pytest.mark.parametrize(
    'shape, broadcast_shape, index, expected_shape',
    [
        ((1, 6, 7), (5, 6, 7), (slice(None), torch.ones(4, 5).int(), slice(None)), (4, 5, 1, 1, 7)),  # array index
        ((1, 6, 7), (5, 6, 7), (slice(None), slice(None), slice(None)), (1, 6, 7)),  # nothing
        ((1, 6, 7), (5, 6, 7), (), (1, 6, 7)),  # nothing
        ((5, 6, 1), (5, 6, 7), (1,), (1, 6, 1)),  # integer indexing
        ((5, 1, 1), (5, 6, 7), (slice(None), 2), (5, 1, 1)),  # integer indexing broadcast
        ((1, 1, 7), (5, 6, 7), (slice(1, 2), slice(None, None, 2), slice(1, None, 2)), (1, 1, 3)),  # slices
        ((5, 1, 1), (5, 6, 7), (torch.tensor([0, 2]),), (2, 1, 1)),  # array index
        ((1, 6, 1), (5, 6, 7), (slice(None), [0, 2]), (1, 2, 1)),  # integer list
        ((1, 1, 1), (5, 6, 7), (torch.tensor([0, 2]), slice(None), slice(None)), (1, 1, 1)),  # array index broadcast
        ((5, 1, 1), (5, 6, 7), (torch.tensor([0, 2]), slice(None), (0, 2)), (2, 1, 1, 1)),  # two array indicces
        ((1, 1, 1), (5, 6, 7), (torch.tensor([0, 2]), slice(None), slice(None)), (1, 1, 1)),  # array index broadcast
        ((5, 6, 7), (5, 6, 7), (slice(None), torch.tensor([1, 3]), slice(None)), (5, 2, 7)),
        ((5, 6, 7), (5, 6, 7), (slice(None), slice(None), torch.tensor([0, 5])), (5, 6, 2)),
        ((5, 6, 7), (5, 6, 7), (torch.tensor([True, False, True, False, True]), slice(None), slice(None)), (3, 6, 7)),
        ((5, 6, 7), (5, 6, 7), (torch.ones(5, 6, 7).bool(),), (210, 1, 1, 1)),
        (
            (5, 6, 7),
            (5, 6, 7),
            (torch.tensor([True, False, True, True, False]), slice(None), slice(None)),
            (3, 6, 7),
        ),
        ((5, 6, 7), (5, 6, 7), (torch.ones(5).bool(), ...), (5, 6, 7)),
        ((5, 6, 7), (5, 6, 7), (torch.tensor([[0, 1], [2, 3]]), slice(None), slice(None)), (2, 2, 1, 6, 7)),
        ((5, 1, 7), (5, 6, 7), (slice(None), torch.tensor([[1, 2], [3, 4]]), slice(None)), (1, 1, 5, 1, 7)),
        ((5, 6, 7), (5, 6, 7), (slice(None), torch.tensor([True, False, True, True, False, True]), 0), (5, 4, 1)),
        ((5, 6, 7), (5, 6, 7), (torch.tensor([1, 3]), slice(None), torch.tensor([0, 5])), (2, 1, 6, 1)),
        ((5, 6, 7), (5, 6, 7), (None, slice(None), slice(None)), (1, 5, 6, 7)),
        ((5, 6, 7), (5, 6, 7), (None, None, slice(None), slice(None)), (1, 1, 5, 6, 7)),
        ((5, 6, 7), (5, 6, 7), (..., 0), (5, 6, 1)),
        ((5, 6, 7), (5, 6, 7), (slice(None), ..., slice(None)), (5, 6, 7)),
        ((5, 6, 7), (5, 6, 7), (1, ..., 3), (1, 6, 1)),
        ((5, 1, 7), (5, 6, 7), (torch.ones(1, 1, 1, dtype=torch.bool),), (5, 1, 7)),
        ((5, 1, 1), (5, 6, 7), (torch.ones(1, 6, 1, dtype=torch.bool),), (5, 1, 1)),
    ],
)
def test_indexer(shape, broadcast_shape, index, expected_shape):
    tensor = torch.arange(int(torch.prod(torch.tensor(shape)))).reshape(shape)
    indexer = Indexer(broadcast_shape, index)
    result = indexer(tensor)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ('index', 'error_message'),
    [
        # Type errors
        ('invalid_index', 'Unsupported index type'),
        (torch.tensor([1.0]), 'Unsupported index dtype'),
        ((Ellipsis, Ellipsis), 'Only one ellipsis is allowed'),
        ((slice(None), None), 'New axes are only allowed at the beginning'),
        ((slice(None, None, -1),), 'Negative step size for slices is not supported'),
        ((0, 1, 2, 3, 4, 5), 'Too many indices'),
        ((5,), 'Index 5 out of bounds'),
        ((torch.tensor([10]),), 'Index out of bounds'),
        (([10],), r'Index out of bounds. Got values in the interval \[10, 11\)'),
        ((torch.ones(3, 1, 6, dtype=torch.bool),), 'Boolean index has wrong shape'),
        (([0, 1], [0, 1, 2]), 'All vectorized indices must have the same shape'),
        (([0, 1], torch.zeros(2, 1).int()), 'All vectorized indices must have the same shape'),
        ((torch.tensor([True, False]), torch.tensor([True, False])), 'Only one boolean index is allowed'),
    ],
)
def test_indexer_invalid_indexing(index, error_message):
    """Test various invalid indexing scenarios."""
    shape = (3, 4, 5)
    with pytest.raises(IndexError, match=error_message):
        Indexer(shape, index)


def test_indexer_broadcast_error():
    """Test error when tensor cannot be broadcast to target shape."""
    shape = (3, 4, 5)
    tensor = torch.arange(24).reshape(2, 3, 4)
    indexer = Indexer(shape, (slice(None),) * 3)

    with pytest.raises(IndexError, match='cannot be broadcasted'):
        indexer(tensor)
