import pytest
import torch
from mrpro.operators.models import MOLLI
from mrpro.operators.models import InversionRecovery
from mrpro.operators.models import SaturationRecovery
from tests import RandomGenerator

SHAPE_VARIATIONS = pytest.mark.parametrize(
    ('parameter_shape', 'ti_shape', 'signal_shape'),
    [
        ((1, 1, 10, 20, 30), (5), (5, 1, 1, 10, 20, 30)),  # single map with different inversion times
        ((1, 1, 10, 20, 30), (5, 1), (5, 1, 1, 10, 20, 30)),
        ((4, 1, 1, 10, 20, 30), (5, 1), (5, 4, 1, 1, 10, 20, 30)),  # multiple maps along additional batch dimension
        ((4, 1, 1, 10, 20, 30), (5), (5, 4, 1, 1, 10, 20, 30)),
        ((4, 1, 1, 10, 20, 30), (5, 4), (5, 4, 1, 1, 10, 20, 30)),
        ((3, 1, 10, 20, 30), (5), (5, 3, 1, 10, 20, 30)),  # multiple maps along other dimension
        ((3, 1, 10, 20, 30), (5, 1), (5, 3, 1, 10, 20, 30)),
        ((3, 1, 10, 20, 30), (5, 3), (5, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5), (5, 4, 3, 1, 10, 20, 30)),  # multiple maps along other and batch dimension
        ((4, 3, 1, 10, 20, 30), (5, 4), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 4, 1), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 1, 3), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 4, 3), (5, 4, 3, 1, 10, 20, 30)),
        ((1,), (5), (5, 1)),  # single voxel
        ((4, 3, 1), (5, 4, 3), (5, 4, 3, 1)),
    ],
)


def create_parameter_tensors(data_shape=(10, 5, 100, 100, 100), number_of_tensors=2):
    random_generator = RandomGenerator(seed=0)
    parameter_tensors = random_generator.float32_tensor(size=(number_of_tensors, *data_shape), low=1e-10)
    return torch.unbind(parameter_tensors)


@SHAPE_VARIATIONS
def test_saturation_recovery_shape(parameter_shape, ti_shape, signal_shape):
    """Test correct signal shapes."""
    random_generator = RandomGenerator(seed=0)
    ti = random_generator.float32_tensor(size=ti_shape, low=0)
    model = SaturationRecovery(ti)
    m0, t1 = create_parameter_tensors(parameter_shape)
    (signal,) = model.forward(m0, t1)
    assert signal.shape == signal_shape


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


@SHAPE_VARIATIONS
def test_inversion_recovery_shape(parameter_shape, ti_shape, signal_shape):
    """Test correct signal shapes."""
    random_generator = RandomGenerator(seed=0)
    ti = random_generator.float32_tensor(size=ti_shape, low=0)
    model = InversionRecovery(ti)
    m0, t1 = create_parameter_tensors(parameter_shape)
    (signal,) = model.forward(m0, t1)
    assert signal.shape == signal_shape


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


@SHAPE_VARIATIONS
def test_molli_shape(parameter_shape, ti_shape, signal_shape):
    """Test correct signal shapes."""
    random_generator = RandomGenerator(seed=0)
    ti = random_generator.float32_tensor(size=ti_shape, low=0)
    model = MOLLI(ti)
    a, b, t1 = create_parameter_tensors(parameter_shape, number_of_tensors=3)
    (signal,) = model.forward(a, b, t1)
    assert signal.shape == signal_shape


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
