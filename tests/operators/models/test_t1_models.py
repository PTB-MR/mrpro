import pytest
import torch
from mrpro.operators.models import MOLLI
from mrpro.operators.models import InversionRecovery
from mrpro.operators.models import SaturationRecovery
from tests.operators.models.test_shape_all_models import create_parameter_tensors


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_saturation_recovery(ti, result):
    """Test for saturation recovery.

    Checking that idata output tensor at ti=0 is close to 0. Checking
    that idata output tensor at large ti is close to m0.
    """
    model = SaturationRecovery(ti)
    m0, t1 = create_parameter_tensors()
    (image,) = model.forward(m0, t1)

    zeros = torch.zeros_like(m0)

    # Assert closeness to zero for ti=0
    if result == '0':
        torch.testing.assert_close(image[0, ...], zeros)
    # Assert closeness to m0 for large ti
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, '-m0'),  # short ti
        (20, 'm0'),  # long ti
    ],
)
def test_inversion_recovery(ti, result):
    """Test for inversion recovery.

    Checking that idata output tensor at ti=0 is close to -m0. Checking
    that idata output tensor at large ti is close to m0.
    """
    model = InversionRecovery(ti)
    m0, t1 = create_parameter_tensors()
    (image,) = model.forward(m0, t1)

    # Assert closeness to -m0 for ti=0
    if result == '-m0':
        torch.testing.assert_close(image[0, ...], -m0)
    # Assert closeness to m0 for large ti
    elif result == 'm0':
        torch.testing.assert_close(image[0, ...], m0)


@pytest.mark.parametrize(
    ('ti', 'result'),
    [
        (0, 'a-b'),  # short ti
        (20, 'a'),  # long ti
    ],
)
def test_molli(ti, result):
    """Test for MOLLI.

    Checking that idata output tensor at ti=0 is close to a. Checking
    that idata output tensor at large ti is close to a-b.
    """
    # Generate qdata tensor, not random as a<b is necessary for t1_star to be >= 0
    other, coils, z, y, x = 10, 5, 100, 100, 100
    a = torch.ones((other, coils, z, y, x)) * 2
    b = torch.ones((other, coils, z, y, x)) * 5
    t1 = torch.ones((other, coils, z, y, x)) * 2

    # Generate signal model and torch tensor for comparison
    model = MOLLI(ti)
    (image,) = model.forward(a, b, t1)

    # Assert closeness to a-b for large ti
    if result == 'a-b':
        torch.testing.assert_close(image[0, ...], a - b)
    # Assert closeness to a for ti=0
    elif result == 'a':
        torch.testing.assert_close(image[0, ...], a)
