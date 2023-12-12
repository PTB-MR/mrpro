import pytest
import torch

from mrpro.data import KTrajectory
from mrpro.data import SpatialDimension
from mrpro.operators.models.WASABI import WASABI
from tests import RandomGenerator


def create_data(offset_max=250, offset_nr=101, b0_shift=0, rb1=1.0, c=1.0, d=2.0, p=4, other=1, coils=1, z=1, y=1, x=1):
    offsets = torch.linspace(-offset_max, offset_max, offset_nr)
    qdata = torch.ones(p, other, coils, z, y, x)
    qdata[0, ...] = b0_shift
    qdata[1, ...] = rb1
    qdata[2, ...] = c
    qdata[3, ...] = d

    return offsets, qdata


@pytest.mark.parametrize(
    'offset_max, offset_nr, b0_shift, rb1, c, d, p, other, coils, z, y, x',
    [
        (250, 101, 0, 1.0, 1.0, 2.0, 4, 1, 1, 1, 1, 1),
        (200, 101, 0, 0, 1.0, 2.0, 4, 1, 1, 1, 1, 1),
        (10, 101, 10, 10, 1.0, 2.0, 4, 1, 1, 1, 1, 1),
        (200, 101, 0, 0, 1.0, 2.0, 4, 1, 1, 1, 1, 1),
    ],
)
def test_WASABI_signal_model(offset_max, offset_nr, b0_shift, rb1, c, d, p, other, coils, z, y, x):
    offsets, qdata = create_data(offset_max, offset_nr, b0_shift, rb1, c, d, p, other, coils, z, y, x)
    wasabi_model = WASABI(offsets=offsets)
    sig = wasabi_model.forward(qdata)

    signal_shape = torch.Tensor(offset_nr, coils, z, y, x)
    assert sig.shape == signal_shape.shape
