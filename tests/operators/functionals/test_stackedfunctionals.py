import pytest
import torch
from mrpro.operators import Functional, ProximableFunctional
from mrpro.operators.Functional import StackedFunctionals, StackedProximableFunctionals

from tests import RandomGenerator


class Dummy(Functional):
    """Only for testing purposes."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x.sum() + 1,)


class ProxDummy(ProximableFunctional):
    """Only for testing purposes."""

    def __init__(self, prox_scale: float = 1.0):
        super().__init__()
        self.register_buffer('dummy', torch.tensor(1.0))
        self.prox_scale = prox_scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x.max() + self.dummy,)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """This is not a real proximal operator, just a dummy function."""
        return (self.prox_scale * x * sigma,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """This is not a real proximal operator, just a dummy function."""

        return (-self.prox_scale * x * sigma,)


def test_stackedfunctions_forward():
    a = Dummy()
    b = ProxDummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    stacked = StackedFunctionals(a, b)
    torch.testing.assert_close(stacked(x1, x2)[0], a(x1)[0] + b(x2)[0])


@pytest.mark.cuda()
def test_stackedfunctions_forward_cuda():
    a = Dummy()
    b = ProxDummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    stacked = StackedFunctionals(a, b)
    torch.testing.assert_close(stacked(x1, x2)[0], a(x1)[0] + b(x2)[0])


def test_stackedfunctions_shorthand():
    a = Dummy()
    b = ProxDummy()

    rng = RandomGenerator(123)
    x1 = rng.float64_tensor(3)
    x2 = rng.float64_tensor(3)
    x3 = rng.float64_tensor(3)
    x4 = rng.float64_tensor(3)

    stacked = a | b
    # mypy will complain about the type of the expression and the argument not matching
    # if the types hints in StackedFunctional are not correct.
    mypy_test_dummy1: StackedFunctionals[torch.Tensor, torch.Tensor] = stacked  # noqa F841
    torch.testing.assert_close(stacked(x1, x2)[0], a(x1)[0] + b(x2)[0])

    stacked2left = b | stacked
    mypy_test_dummy2: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor] = stacked2left  # noqa F841
    torch.testing.assert_close(stacked2left(x1, x2, x3)[0], b(x1)[0] + a(x2)[0] + b(x3)[0])

    stacked2right = stacked | b
    mypy_test_dummy3: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor] = stacked2left  # noqa F841
    torch.testing.assert_close(stacked2right(x1, x2, x3)[0], b(x1)[0] + a(x2)[0] + b(x3)[0])

    stacked3 = stacked | stacked
    mypy_test_dummy4: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = stacked3  # noqa F841
    torch.testing.assert_close(stacked3(x1, x2, x3, x4)[0], a(x1)[0] + b(x2)[0] + a(x3)[0] + +b(x4)[0])


def test_stackedfunctionals_prox():
    rng = RandomGenerator(123)
    xs = rng.float32_tensor(4)
    stacked = ProxDummy(1.0) | (ProxDummy(2.0) | ProxDummy(3.0)) | ProxDummy(4.0)
    expected_prox = tuple(torch.tensor([1.0, 2.0, 3.0, 4.0]) * xs)
    actual_prox = stacked.prox(*xs, sigma=1.0)
    assert expected_prox == pytest.approx(actual_prox)
    # mypy will complain about the type of the expression and the argument not matching
    # if the types hints in StackedProximableFunctionals are not correct.
    mypy_test_dummy1: StackedProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = stacked  # noqa F841
    mypy_test_dummy2: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = actual_prox  # noqa F841


def test_stackedfunctionals_prox_convex_conj():
    rng = RandomGenerator(123)
    xs = rng.float32_tensor(4)
    stacked = ProxDummy(1.0) | (ProxDummy(2.0) | ProxDummy(3.0)) | ProxDummy(4.0)
    expected_proxcc = tuple(torch.tensor([1.0, 2.0, 3.0, 4.0]) * xs)
    actual_prox = stacked.prox_convex_conj(*xs, sigma=-1.0)
    assert expected_proxcc == pytest.approx(actual_prox)
    # mypy will complain about the type of the expression and the argument not matching
    # if the types hints in StackedProximableFunctionals are not correct.
    mypy_test_dummy1: StackedProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = stacked  # noqa F841
    mypy_test_dummy2: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] = actual_prox  # noqa F841


def test_stackedfunctionals_iter_order():
    a = Dummy()
    b = ProxDummy()
    c = ProxDummy()
    assert tuple(a | b) == (a, b)
    assert tuple(b | c) == (b, c)
    assert tuple((b | c) | (b | c) | b) == (b, c, b, c, b)
    assert tuple((a | c) | (b | c) | b) == (a, c, b, c, b)


def test_stackedfunctionls_len():
    a = Dummy()
    b = ProxDummy()
    assert len(a | b) == 2
    assert len(b | b | b) == 3
