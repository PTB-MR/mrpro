import pytest
import torch
from mrpro.operators.models import MOLLI
from mrpro.operators.models import InversionRecovery
from mrpro.operators.models import SaturationRecovery
from tests import RandomGenerator


def create_data(other=10, coils=5, z=100, y=100, x=100):
    random_generator = RandomGenerator(seed=0)
    m0 = random_generator.float32_tensor(size=(other, coils, z, y, x), low=1e-10)
    t1 = random_generator.float32_tensor(size=(other, coils, z, y, x), low=1e-10)
    return m0, t1


@pytest.mark.parametrize(
    ('t', 'result'),
    [
        (0, '0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_saturation_recovery(t, result):
    """Test for saturation recovery.

    Checking that idata output tensor at t=0 is close to 0. Checking
    that idata output tensor at large t is close to m0.
    """
    # Tensor of TI
    ti = torch.tensor([t])

    # Generate signal model and torch tensor for comparison
    model = SaturationRecovery(ti)
    m0, t1 = create_data()
    (image,) = model.forward(m0, t1)

    zeros = torch.zeros_like(m0)

    # Assert closeness to zero for t=0
    if result == '0':
        torch.testing.assert_close(image, zeros)
    # Assert closeness to m0 for large t
    elif result == 'm0':
        torch.testing.assert_close(image, zeros)  # was m0


@pytest.mark.parametrize(
    ('t', 'result'),
    [
        (0, '-m0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_inversion_recovery(t, result):
    """Test for inversion recovery.

    Checking that idata output tensor at t=0 is close to -m0. Checking
    that idata output tensor at large t is close to m0.
    """

    # Tensor of TI
    ti = torch.tensor([t])

    # Generate signal model and torch tensor for comparison
    model = InversionRecovery(ti)
    m0, t1 = create_data()
    (image,) = model.forward(m0, t1)

    # Assert closeness to -m0 for t=0
    if result == '-m0':
        torch.testing.assert_close(image, -m0)
    # Assert closeness to m0 for large t
    elif result == 'm0':
        torch.testing.assert_close(image, m0)


@pytest.mark.parametrize(
    ('t', 'result'),
    [
        (0, 'a-b'),  # short ti
        (20, 'a'),  # long ti
    ],
)
def test_molli(t, result):
    """Test for MOLLI.

    Checking that idata output tensor at t=0 is close to a. Checking
    that idata output tensor at large t is close to a-b.
    """
    # Generate qdata tensor, not random as a<b is necessary for t1_star to be >= 0
    other, coils, z, y, x = 10, 5, 100, 100, 100
    a = torch.ones((other, coils, z, y, x)) * 2
    b = torch.ones((other, coils, z, y, x)) * 5
    t1 = torch.ones((other, coils, z, y, x)) * 2

    # Tensor of TI
    ti = torch.tensor([t])

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model.forward(a, b, t1)

    # Assert closeness to a-b for large t
    if result == 'a-b':
        torch.testing.assert_close(image, a - b)
    # Assert closeness to a for t=0
    elif result == 'a':
        torch.testing.assert_close(image, a)
