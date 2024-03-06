import pytest
import torch
from mrpro.operators.models._WASABI import WASABI


def create_data(offset_max=250, offset_nr=101, b0_shift_in=0, rb1=1.0, c=1.0, d=2.0):
    offsets = torch.linspace(-offset_max, offset_max, offset_nr)
    b0_shift = torch.zeros([1, 1, 1, 1, 1])
    b0_shift[0] = b0_shift_in
    rb1 = torch.Tensor([rb1])
    c = torch.Tensor([c])
    d = torch.Tensor([d])

    return offsets, b0_shift, rb1, c, d


@pytest.mark.parametrize(
    ('offset_max', 'offset_nr', 'b0_shift', 'rb1', 'c', 'd', 'coils', 'z', 'y', 'x'),
    [
        (250, 101, 0, 1.0, 1.0, 2.0, 1, 1, 1, 1),
        (200, 101, 0, 0, 1.0, 2.0, 1, 1, 1, 1),
        (10, 101, 10, 10, 1.0, 2.0, 1, 1, 1, 1),
    ],
)
def test_WASABI_signal_model_shape(offset_max, offset_nr, b0_shift, rb1, c, d, coils, z, y, x):
    """Test for correct output shape."""
    offsets, b0_shift, rb1, c, d = create_data(offset_max, offset_nr, b0_shift, rb1, c, d)
    wasabi_model = WASABI(offsets=offsets)
    signal, = wasabi_model.forward(b0_shift, rb1, c, d)

    assert signal.shape == (offset_nr, coils, z, y, x)


def test_WASABI_shift():
    """Test symmetry property of shifted and unshifted WASABI spectra."""
    offsets_unshifted, b0_shift_unshifted, rb1_unshifted, c_unshifted, d_unshifted = create_data(
        offset_max=500,
        offset_nr=100,
        b0_shift_in=0,
    )
    wasabi_model = WASABI(offsets=offsets_unshifted)
    signal, = wasabi_model.forward(b0_shift_unshifted, rb1_unshifted, c_unshifted, d_unshifted)

    offsets_shifted, b0_shift, rb1, c, d = create_data(offset_max=500, offset_nr=101, b0_shift_in=100)
    wasabi_model = WASABI(offsets=offsets_shifted)
    signal_shifted, = wasabi_model.forward(b0_shift, rb1, c, d)

    lower_index = (offsets_shifted == -300).nonzero()[0][0].item()
    upper_index = (offsets_shifted == 500).nonzero()[0][0].item()

    assert signal[0] == signal[-1], "Result should be symmetric around center"
    assert signal_shifted[lower_index] == signal_shifted[upper_index], "Result should be symmetric around shift"


def test_WASABI_extreme_offset():
    offset, b0_shift, rb1, c, d = create_data(offset_max=30000, offset_nr=1, b0_shift_in=0)
    wasabi_model = WASABI(offsets=offset)
    signal, = wasabi_model.forward(b0_shift, rb1, c, d)

    assert torch.isclose(signal, torch.tensor([1.])), "For an extreme offset, the signal should be unattenuated"
