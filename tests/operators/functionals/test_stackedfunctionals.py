import pytest
import torch
from mrpro.operators import Functional, ProximableFunctional
from mrpro.operators.StackedFunctionals import StackedFunctionals

from tests import RandomGenerator


class Dummy(Functional):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x.sum() + 1,)


class ProxDummy(ProximableFunctional):
    def __init__(self):
        super().__init__()
        self.register_buffer('dummy', torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        return (x.max() + self.dummy,)

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        return (x,)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        return (x,)


def test_stackedfunctions_forward():
    functional1 = Dummy()
    functional2 = ProxDummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    stacked = StackedFunctionals(functional1, functional2)
    torch.testing.assert_close(stacked(x1, x2), functional1(x1) + functional2(x2))


@pytest.mark.cuda()
def test_stackedfunctions_forward_cuda():
    functional1 = Dummy()
    functional2 = ProxDummy()
    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    stacked = StackedFunctionals(functional1, functional2)
    torch.testing.assert_close(stacked(x1, x2), functional1(x1) + functional2(x2))


def test_stackedfunctions_shorthand():
    functional1 = Dummy()
    functional2 = ProxDummy()

    rng = RandomGenerator(123)
    x1 = rng.float32_tensor(3)
    x2 = rng.float32_tensor(3)
    x3 = rng.float32_tensor(3)
    x4 = rng.float32_tensor(3)

    stacked = functional1 | functional2
    mypy_test: StackedFunctionals[torch.Tensor, torch.Tensor] = stacked
    torch.testing.assert_close(stacked(x1, x2), functional1(x1) + functional2(x2))

    stacked2left = functional1 | stacked

    stacked2right = stacked | functional1

    stacked3 = stacked2left | stacked2right
