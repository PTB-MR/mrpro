import pytest
import torch
from mrpro.operators import ProximableFunctional, ProximableFunctionalSeparableSum

from tests import RandomGenerator


class Dummy(ProximableFunctional):
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


def test_separablesum_forward():
    a = Dummy()
    b = Dummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    summed = ProximableFunctionalSeparableSum(a, b)
    torch.testing.assert_close(summed(x1, x2)[0], a(x1)[0] + b(x2)[0])


@pytest.mark.cuda
def test_separablesum_forward_cuda():
    a = Dummy()
    b = Dummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    summed = ProximableFunctionalSeparableSum(a, b)
    torch.testing.assert_close(summed(x1, x2)[0], a(x1)[0] + b(x2)[0])


def test_separablesum_shorthand():
    a = Dummy()
    b = Dummy()

    rng = RandomGenerator(123)
    x1 = rng.float64_tensor(3)
    x2 = rng.float64_tensor(3)
    x3 = rng.float64_tensor(3)
    x4 = rng.float64_tensor(3)

    summed = a | b
    torch.testing.assert_close(summed(x1, x2)[0], a(x1)[0] + b(x2)[0])

    summed2left = b | (a | b)
    torch.testing.assert_close(summed2left(x1, x2, x3)[0], b(x1)[0] + a(x2)[0] + b(x3)[0])

    summed2right = (a | b) | b
    torch.testing.assert_close(summed2right(x1, x2, x3)[0], a(x1)[0] + b(x2)[0] + b(x3)[0])

    summed3 = (a | b) | (b | a)
    torch.testing.assert_close(summed3(x1, x2, x3, x4)[0], a(x1)[0] + b(x2)[0] + b(x3)[0] + a(x4)[0])


def test_ProximableFunctionalSeparableSum_prox():
    rng = RandomGenerator(123)
    xs = rng.float32_tensor(4)
    summed = Dummy(1.0) | (Dummy(2.0) | Dummy(3.0)) | Dummy(4.0)
    expected_prox = tuple(torch.tensor([1.0, 2.0, 3.0, 4.0]) * xs)
    actual_prox = summed.prox(*xs, sigma=1.0)
    assert expected_prox == pytest.approx(actual_prox)


def test_ProximableFunctionalSeparableSum_prox_convex_conj():
    rng = RandomGenerator(123)
    xs = rng.float32_tensor(4)
    summed = Dummy(1.0) | (Dummy(2.0) | Dummy(3.0)) | Dummy(4.0)
    expected_proxcc = tuple(torch.tensor([1.0, 2.0, 3.0, 4.0]) * xs)
    actual_prox = summed.prox_convex_conj(*xs, sigma=-1.0)
    assert expected_proxcc == pytest.approx(actual_prox)


def test_ProximableFunctionalSeparableSum_iter_order():
    a = Dummy()
    b = Dummy()
    c = Dummy()
    assert tuple(a | b) == (a, b)
    assert tuple(b | c) == (b, c)
    assert tuple((b | c) | (b | c) | b) == (b, c, b, c, b)
    assert tuple((a | c) | (b | c) | b) == (a, c, b, c, b)


def test_summedfunctionls_len():
    a = Dummy()
    b = Dummy()
    assert len(a | b) == 2
    assert len(b | b | b) == 3
