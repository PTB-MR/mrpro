from typing import Literal, cast

import pytest
import torch
from mrpro.operators import ElementaryFunctional, ElementaryProximableFunctional, ProximableFunctional, ScaledFunctional

from tests import RandomGenerator
from tests.operators.functionals.conftest import (
    FUNCTIONALS,
    PROXIMABLE_FUNCTIONALS,
    FunctionalTestCase,
    functional_test_cases,
)


@pytest.mark.parametrize('functional', FUNCTIONALS)
@pytest.mark.parametrize('scale_type', ['negative', 'positive', 'tensor', 'int'])
def test_functional_scaling_forward(
    functional: type[ElementaryFunctional], scale_type: Literal['negative', 'positive', 'tensor', 'int']
):
    """Test if forward method is scaled."""
    rng = RandomGenerator(1)
    x = rng.complex64_tensor((10, 10))
    f = functional()
    match scale_type:
        case 'negative':
            scale: float | torch.Tensor = -4.0
        case 'positive':
            scale = 3.0
        case 'tensor':
            scale = rng.float32_tensor()
        case 'int':
            scale = 5
    scaled_f = cast(ProximableFunctional, scale * f)
    assert isinstance(scaled_f, ScaledFunctional)
    if isinstance(f, ProximableFunctional):
        assert isinstance(scaled_f, ProximableFunctional)
    torch.testing.assert_close(scaled_f(x)[0], scale * f(x)[0])


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_negative_scale(
    functional: type[ElementaryProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if error is raised for negative or complex scale"""

    def case(scale):
        x = torch.zeros(2)
        with pytest.raises((ValueError, TypeError)):
            getattr(scale * functional(keepdim=True), function)(x)

    # all these cases should throw an exception:
    case(scale=-1)  # negative
    case(scale=1j)  # complex
    case(scale=torch.tensor((1, -1)))  # one element negative
    case(scale=torch.tensor(1.0, dtype=torch.complex64))  # complex dtype


@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_scaling_zero_forward(functional: type[ElementaryFunctional]):
    """Test if zero scaled forward is zero."""
    rng = RandomGenerator(1)
    x = rng.complex64_tensor((10, 10))
    f = cast(ProximableFunctional, torch.zeros(10) * functional(dim=1))
    (fx,) = f(x)
    torch.testing.assert_close(fx, 0 * fx)


@functional_test_cases
def test_functional_zero_scaling_prox(case: FunctionalTestCase):
    """Test if prox zero scaled prox is identity."""
    x = case.rand_x()
    f = 0.0 * case.functional
    (prox,) = f.prox(x)
    torch.testing.assert_close(prox, x.to(case.result_dtype))


@functional_test_cases
def test_functional_scaling_moreau(case: FunctionalTestCase):
    """Test if Moreau identity holds if scaled."""
    scale = case.sigma
    functional = cast(ProximableFunctional, scale * case.functional)
    x = case.rand_x()
    (prox,) = functional.prox(x)
    (prox_convex_conj,) = functional.prox_convex_conj(x)
    x_new = prox + prox_convex_conj
    torch.testing.assert_close(x.to(case.result_dtype), x_new, rtol=1e-3, atol=1e-3)


@functional_test_cases
def test_functional_scaling_prox_optimality(case: FunctionalTestCase):
    """Use autograd to check if prox criterion is minimized."""
    x = case.rand_x()
    scale = case.rng.float32(1, 2)
    functional = case.functional
    scaled_functional = scale * functional

    (prox,) = scaled_functional.prox(x, sigma=case.sigma)

    def prox_criterion(p):
        diff = x - p
        l2 = torch.sum((diff * diff.conj()).real, dim=functional.dim, keepdim=functional.keepdim)
        return (scaled_functional(p)[0] + 1 / (2 * case.sigma) * l2).sum()

    for perturbation in (0, 1e-3, 0.1):
        p = (prox + perturbation * case.rng.rand_like(prox)).requires_grad_()
        optim = torch.optim.SGD([p], lr=1e-2)
        for _ in range(200):
            optim.zero_grad()
            prox_criterion(p).backward()
            optim.step()
            optim.param_groups[0]['lr'] *= 0.97

        assert prox_criterion(p) + 1e-5 >= prox_criterion(prox)
