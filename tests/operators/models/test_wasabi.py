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
    sig = wasabi_model.forward(b0_shift, rb1, c, d)

    signal_shape = torch.Tensor(offset_nr, coils, z, y, x)
    assert sig.shape == signal_shape.shape


def test_WASABI_shift():
    """Test symmetry property of shifted and unshifted WASABITI spectra."""
    offsets_unshifted, b0_shift_unshifted, rb1_unshifted, c_unshifted, d_unshifted = create_data(
        offset_max=500,
        offset_nr=100,
        b0_shift_in=0,
    )
    wasabi_model = WASABI(offsets=offsets_unshifted)
    sig_unshifted = wasabi_model.forward(b0_shift_unshifted, rb1_unshifted, c_unshifted, d_unshifted)

    offsets_shifted, b0_shift, rb1, c, d = create_data(offset_max=500, offset_nr=101, b0_shift_in=100)
    wasabi_model = WASABI(offsets=offsets_shifted)
    sig_shifted = wasabi_model.forward(b0_shift, rb1, c, d)

    lower_index = (offsets_shifted == -300).nonzero()[0][0].item()
    upper_index = (offsets_shifted == 500).nonzero()[0][0].item()

    assert sig_unshifted[0] == sig_unshifted[-1]
    assert sig_shifted[lower_index] == sig_shifted[upper_index]


def test_WASABI_symmetry():
    _, b0_shift, rb1, c, d = create_data(offset_max=300, offset_nr=3, b0_shift_in=0)
    offsets_new = torch.FloatTensor([30000, 500, 0, 200, 5000])
    wasabi_model = WASABI(offsets=offsets_new)
    sig = wasabi_model.forward(b0_shift, rb1, c, d)

    assert torch.isclose(sig[0], torch.FloatTensor([1]), rtol=1e-8)
