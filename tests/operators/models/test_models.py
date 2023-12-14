import pytest
import torch

from mrpro.operators.models._InversionRecovery import InversionRecovery
from mrpro.operators.models._MOLLI import MOLLI
from mrpro.operators.models._SaturationRecovery import SaturationRecovery
from tests import RandomGenerator


def create_data(p, other=10, coils=5, z=100, y=100, x=100):
    random_generator = RandomGenerator(seed=0)
    qdata = random_generator.float32_tensor(size=(p, other, coils, z, y, x), low=1e-10)
    return qdata


@pytest.mark.parametrize(
    't, result',
    [
        # short ti
        (0, '0'),
        # long ti
        (20, 'm0'),
    ],
)
def test_saturation_recovery(t, result):
    """Test for saturation recovery.

    Checking that idata output tensor at t=0 is close to 0. Checking
    that idata output tensor at large t is close to m0.
    """

    # Random qdata tensor
    p, other, coils, z, y, x = 2, 10, 5, 100, 100, 100
    qdata = create_data(p, other, coils, z, y, x)

    # Tensor of TI
    ti = torch.tensor([t])

    # Generate signal model and torch tensor for comparison
    model = SaturationRecovery(ti)
    image = model.forward(qdata)

    zeros = torch.zeros(size=((len(ti) * other, coils, z, y, x)))
    m0 = qdata[0]

    # Assert closeness to zero for t=0
    if result == '0':
        torch.testing.assert_close(image, zeros)
    # Assert closeness to m0 for large t
    elif result == 'm0':
        torch.testing.assert_close(image, m0)


@pytest.mark.parametrize(
    't, result',
    [
        # short ti
        (0, '-m0'),
        # long ti
        (20, 'm0'),
    ],
)
def test_inversion_recovery(t, result):
    """Test for inversion recovery.

    Checking that idata output tensor at t=0 is close to -m0. Checking
    that idata output tensor at large t is close to m0.
    """

    # Random qdata tensor
    p, other, coils, z, y, x = 2, 10, 5, 100, 100, 100
    qdata = create_data(p, other, coils, z, y, x)

    # Tensor of TI
    ti = torch.tensor([t])

    # Generate signal model and torch tensor for comparison
    model = InversionRecovery(ti)
    image = model.forward(qdata)
    m0 = qdata[0]

    # Assert closeness to -m0 for t=0
    if result == '-m0':
        torch.testing.assert_close(image, -m0)
    # Assert closeness to m0 for large t
    elif result == 'm0':
        torch.testing.assert_close(image, m0)


@pytest.mark.parametrize(
    't, result',
    [
        # short ti
        (0, 'a-b'),
        # long ti
        (20, 'a'),
    ],
)
def test_molli(t, result):
    """Test for MOLLI.

    Checking that idata output tensor at t=0 is close to a. Checking
    that idata output tensor at large t is close to a-b.
    """
    # Generate qdata tensor, not random as a<b is necessary for t1_star to be >= 0
    p, other, coils, z, y, x = 3, 10, 5, 100, 100, 100
    qdata = torch.zeros(size=(p, other, coils, z, y, x))
    qdata[0] = torch.ones((1, other, coils, z, y, x)) * 2
    qdata[1] = torch.ones((1, other, coils, z, y, x)) * 5
    qdata[2] = torch.ones((1, other, coils, z, y, x)) * 2

    # Tensor of TI
    ti = torch.tensor([t])

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    image = model.forward(qdata)
    a = qdata[0]
    b = qdata[1]

    # Assert closeness to a-b for large t
    if result == 'a-b':
        torch.testing.assert_close(image, a - b)
    # Assert closeness to a for t=0
    elif result == 'a':
        torch.testing.assert_close(image, a)
