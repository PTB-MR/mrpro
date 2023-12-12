import pytest
import torch

from mrpro.operators.models import InversionRecovery
from mrpro.operators.models import Molli
from mrpro.operators.models import SaturationRecovery
from tests import RandomGenerator


def test_saturation_recovery_t_0():
    """Test for saturation recovery at t=0.

    Checking that idata output tensor is close to 0.
    """

    # Random qdata tensor
    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    other = 10
    random_tensor = random_generator.float32_tensor(size=(2, other, num_coils, Nz, Ny, Nx), low=1e-10)

    # Tensor of TI
    ti = torch.tensor([0.0])

    # Generate signal model and torch tensor for comparison
    model = SaturationRecovery(ti)
    image = model.forward(random_tensor)
    zeros = torch.zeros_like(image)

    # Assert closeness to zero
    torch.testing.assert_close(image, zeros)

    assert image is not None
    assert image.shape == (len(ti) * other, num_coils, Nz, Ny, Nx)


def test_saturation_recovery_t_large():
    """Test for saturation recovery at large t.

    Checking that idata output tensor is close to m0.
    """

    # Random qdata tensor
    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    other = 10
    random_tensor = random_generator.float32_tensor(size=(2, other, num_coils, Nz, Ny, Nx), low=1e-10)

    # Tensor of TI
    ti = torch.tensor([20])

    # Generate signal model and torch tensor for comparison
    model = SaturationRecovery(ti)
    image = model.forward(random_tensor)
    m0 = random_tensor[0]

    # Assert closeness to m0
    torch.testing.assert_close(image, m0)

    assert image is not None
    assert image.shape == (len(ti) * other, num_coils, Nz, Ny, Nx)


def test_inversion_recovery_t_0():
    """Test for inversion recovery at t=0.

    Checking that idata output tensor is close to -m0.
    """

    # Random qdata tensor
    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    other = 10
    random_tensor = random_generator.float32_tensor(size=(2, other, num_coils, Nz, Ny, Nx), low=1e-10)

    # Tensor of TI
    ti = torch.tensor([0.0])

    # Generate signal model and torch tensor for comparison
    model = InversionRecovery(ti)
    image = model.forward(random_tensor)
    m0 = random_tensor[0]

    # Assert closeness to -m0
    torch.testing.assert_close(image, -m0)

    assert image is not None
    assert image.shape == (len(ti) * other, num_coils, Nz, Ny, Nx)


def test_inversion_recovery_t_large():
    """Test for inversion recovery at large t.

    Checking that idata output tensor is close to m0.
    """

    # Random qdata tensor
    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    other = 10
    random_tensor = random_generator.float32_tensor(size=(2, other, num_coils, Nz, Ny, Nx), low=1e-10)

    # Tensor of TI
    ti = torch.tensor([20])

    # Generate signal model and torch tensor for comparison
    model = InversionRecovery(ti)
    image = model.forward(random_tensor)
    m0 = random_tensor[0]

    # Assert closeness to m0
    torch.testing.assert_close(image, m0)

    assert image is not None
    assert image.shape == (len(ti) * other, num_coils, Nz, Ny, Nx)


def test_molli_t_0():
    """Test for molli at t=0.

    Checking that idata output tensor is close to a - b.
    """

    # Random qdata tensor
    random_generator = RandomGenerator(seed=0)
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    other = 10
    random_tensor = random_generator.float32_tensor(size=(3, other, num_coils, Nz, Ny, Nx), low=1e-10)

    # Tensor of TI
    ti = torch.tensor([0.0])
    n = torch.tensor([5])
    rr = torch.tensor([0.0])

    # Generate signal model and torch tensor for comparison
    model = Molli(ti, n, rr)
    image = model.forward(random_tensor)
    a = random_tensor[0]
    b = random_tensor[1]

    # Assert closeness to a-b
    torch.testing.assert_close(image, (a - b))

    assert image is not None
    assert image.shape == (len(ti) * other, num_coils, Nz, Ny, Nx)


def test_molli_t_large():
    """Test for molli at large t.

    Checking that idata output tensor is close to a.
    """

    # Generate qdata tensor, not random as a<b is necessary for t1_star to be >= 0
    Nz, Ny, Nx = 100, 100, 100
    num_coils = 4
    other = 10
    random_tensor = torch.zeros(size=(3, other, num_coils, Nz, Ny, Nx))
    random_tensor[0] = torch.ones((1, other, num_coils, Nz, Ny, Nx)) * 2
    random_tensor[1] = torch.ones((1, other, num_coils, Nz, Ny, Nx)) * 5
    random_tensor[2] = torch.ones((1, other, num_coils, Nz, Ny, Nx)) * 2

    # Tensor of TI
    ti = torch.tensor([20])
    n = torch.tensor([5])
    rr = torch.tensor([0.0])

    # Generate signal model and torch tensor for comparison
    model = Molli(ti, n, rr)
    image = model.forward(random_tensor)
    a = random_tensor[0]

    # Assert closeness to a
    torch.testing.assert_close(image, a)

    assert image is not None
    assert image.shape == (len(ti) * other, num_coils, Nz, Ny, Nx)
