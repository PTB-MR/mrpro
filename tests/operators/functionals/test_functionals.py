import inspect
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
import torch.test
from mrpro.operators import Functional, ProximableFunctional, functionals
from mrpro.operators.functionals import L2NormSquared

from tests import RandomGenerator


@dataclass
class FunctionalTestCase:
    functional: ProximableFunctional
    x_dtype: torch.dtype
    x_shape: torch.Size
    rng: RandomGenerator
    sigma: float | torch.Tensor

    def rand_x(self) -> torch.Tensor:
        return self.rng.rand_tensor(self.x_shape, low=0.0, high=1.0, dtype=self.x_dtype)

    @property
    def result_dtype(self):
        return torch.promote_types(self.x_dtype, torch.result_type(self.functional.target, self.functional.weight))


FUNCTIONALS: list[type[Functional]] = [
    x[1] for x in inspect.getmembers(functionals, lambda x: inspect.isclass(x) and issubclass(x, Functional))
]
PROXIMABLE_FUNCTIONALS: list[type[ProximableFunctional]] = [
    x for x in FUNCTIONALS if issubclass(x, ProximableFunctional)
]


def functional_test_cases(func: Callable[[FunctionalTestCase], None]) -> None:
    @pytest.mark.parametrize('functional', FUNCTIONALS)
    @pytest.mark.parametrize('shape', [[1, 2, 3]])
    @pytest.mark.parametrize('dtype_name', ['float32', 'complex64'])
    @pytest.mark.parametrize('weight', ['scalar_weight', 'tensor_weight', 'binary_weight', 'complex_weight'])
    @pytest.mark.parametrize('target', ['no_target', 'random_target'])
    @pytest.mark.parametrize('dim', [None, -2])
    @pytest.mark.parametrize('divide_by_n', [True, False])
    def wrapper(
        functional: type[ProximableFunctional],
        shape: torch.Size,
        dtype_name: Literal['float32', 'float64', 'complex64', 'complex128'],
        weight: Literal['scalar_weight', 'tensor_weight', 'binary_weight', 'complex_weight'],
        target: Literal['no_target', 'random_target', 'zero_target'],
        dim: None | Sequence[int],
        divide_by_n: bool,
    ):
        dtype = getattr(torch, dtype_name)
        rng = RandomGenerator(13)
        sigma = rng.rand_tensor((1,), low=0.0, high=1.0, dtype=dtype.to_real())
        match weight:
            case 'scalar_weight':
                weight_value: float | torch.Tensor = rng.float64(low=-2, high=2)
            case 'binary_weight':
                weight_value = rng.rand_tensor(shape, low=0.0, high=1.0, dtype=dtype.to_real()).round()
            case 'complex_weight':
                weight_value = rng.rand_tensor(shape, low=0.0, high=1.0, dtype=dtype.to_complex())
            case 'tensor_weight':
                weight_value = rng.rand_tensor(shape, low=-1.0, high=1.0, dtype=dtype.to_real())
        match target:
            case 'zero_target':
                target_value: torch.Tensor | None = torch.zeros(shape, dtype=dtype)
            case 'no_target':
                target_value = None
            case 'random_target':
                target_value = rng.rand_tensor(shape, low=0 if dtype.is_complex else -1, high=1.0, dtype=dtype)
        f = functional(divide_by_n=divide_by_n, weight=weight_value, target=target_value, dim=dim)
        case = FunctionalTestCase(functional=f, x_shape=shape, x_dtype=dtype, rng=rng, sigma=sigma)
        func(case)

    return wrapper


@pytest.mark.parametrize('functional', FUNCTIONALS)
@pytest.mark.parametrize('shape', [[1, 2, 3], ()])
@pytest.mark.parametrize('dim', [None, -2])
@pytest.mark.parametrize('keepdim', [True, False])
def test_functional_shape(functional: type[Functional], shape: torch.Size, dim: None | int, keepdim: bool):
    f = functional(dim=dim, keepdim=keepdim)
    x = torch.ones(shape)

    if dim is not None and (dim >= len(shape) or dim < -len(shape)):
        # dim is out of bounds
        # easier to check here than to remove the cases from the parametrization
        with pytest.raises(IndexError):
            f(x)
        return

    (fx,) = f(x)

    # easier to calculate the expected shape than to parametrize it
    if dim is None and not keepdim:
        # collapse all dimensions, i.e. return a scalar
        expected_shape: tuple[int, ...] = ()
    elif dim is None and keepdim:
        # make all dimensions singleton
        expected_shape = (1,) * len(shape)
    elif not keepdim:
        # remove the dimensions in dim
        expected_shape = tuple([s for i, s in enumerate(shape) if i not in torch.tensor(dim) % len(shape)])
    elif keepdim:
        # make the dimensions in dim singleton
        expected_shape = tuple([s if i not in torch.tensor(dim) % len(shape) else 1 for i, s in enumerate(shape)])
    assert fx.shape == expected_shape


@pytest.mark.parametrize('functional', FUNCTIONALS)
@pytest.mark.parametrize('shape', [[2, 3, 4], ()])
@pytest.mark.parametrize('dim', [None, (0, -2)])
@pytest.mark.parametrize('keepdim', [True, False])
def test_functional_divide_by_n(
    functional: type[Functional], shape: torch.Size, dim: None | tuple[int, ...], keepdim: bool
):
    f_mean = functional(divide_by_n=True, dim=dim, keepdim=keepdim)
    f_sum = functional(divide_by_n=False, dim=dim, keepdim=keepdim)
    x = RandomGenerator(13).float32_tensor(shape)

    if dim is not None and any(d >= len(shape) or d < -len(shape) for d in dim):
        # dim is out of bounds
        # easier to check here than to remove the cases from the parametrization
        with pytest.raises(IndexError):
            f_mean(x)
        with pytest.raises(IndexError):
            f_sum(x)
        return

    ones = torch.ones_like(x)
    n = torch.sum(ones, dim=dim, keepdim=keepdim)
    torch.testing.assert_close(f_mean(x)[0], f_sum(x)[0] / n)


@functional_test_cases
def test_moreau(case: FunctionalTestCase):
    """Test if Moreua identity holds."""
    functional = case.functional
    x = case.rand_x()
    (prox,) = functional.prox(x, sigma=case.sigma)
    (prox_convex_conj,) = functional.prox_convex_conj(x / case.sigma, 1.0 / case.sigma)
    x_new = prox + case.sigma * prox_convex_conj
    torch.testing.assert_close(x.to(case.result_dtype), x_new, rtol=1e-3, atol=1e-3)


@functional_test_cases
def test_prox_optimality(case: FunctionalTestCase):
    """Use autograd to check if prox criterion is minimized."""
    functional = case.functional
    x = case.rand_x()
    (prox,) = functional.prox(x, sigma=case.sigma)
    l2square = L2NormSquared(
        dim=functional.dim,
        keepdim=functional.keepdim,
    )

    def prox_criterion(p):
        return (functional(p)[0] + 1 / (2 * case.sigma) * l2square(p - x)[0]).sum()

    p = prox.clone().detach().requires_grad_()
    optim = torch.optim.SGD([p], lr=1e-2)
    for _ in range(300):
        optim.zero_grad()
        loss = prox_criterion(p)
        loss.backward()
        assert p.grad is not None
        if torch.all(p.grad.abs() < 1e-3):
            break
        optim.step()
        optim.param_groups[0]['lr'] *= 0.98

    torch.testing.assert_close(p, prox, rtol=1e-3, atol=1e-4)


@functional_test_cases
def test_functional_shift(case: FunctionalTestCase):
    """Test if target shift is correct."""
    # translation property
    # \mathrm{prox}_{\sigma F( \cdot - s)}(x) = s + \mathrm{prox}_{\sigma F}(x - s)
    shift = case.rand_x()
    x = case.rand_x()
    functional = case.functional
    prox_target = functional.prox(x - shift, sigma=case.sigma)[0] + shift
    value_target = functional(x - shift)[0]
    shifted_functional = deepcopy(functional)
    shifted_functional.target = functional.target + shift
    prox_shifted = shifted_functional.prox(x, sigma=case.sigma)[0]
    value_shifted = shifted_functional(x)[0]
    torch.testing.assert_close(prox_shifted, prox_target)
    torch.testing.assert_close(value_shifted, value_target)


@functional_test_cases
def test_functional_scaling(case: FunctionalTestCase):
    """Test if scaling with scalar weight is correct"""
    # scaling property
    # \mathrm{prox}_{\sigma F(\alpha \, \cdot)}(x) = \frac{1}{\alpha} \mathrm{prox}_{\sigma \alpha^2 F(\cdot) }(\alpha x)
    alpha = case.rng.float32_tensor(1, 1, 2)
    functional = case.functional
    x = case.rand_x()
    fx_target = functional(alpha * x)
    prox_target = 1 / alpha * functional.prox(alpha * x, sigma=alpha**2 * case.sigma)[0]
    scaled_functional = deepcopy(functional)
    scaled_functional.weight = functional.weight * alpha
    scaled_functional.target = functional.target / alpha
    fx_scaled = scaled_functional(x)
    prox_scaled = scaled_functional.prox(x, sigma=case.sigma)[0]
    torch.testing.assert_close(prox_scaled, prox_target)
    torch.testing.assert_close(fx_scaled, fx_target)


@functional_test_cases
def test_prox_zero_scaling(case: FunctionalTestCase):
    """Test if scaling with scalar zero is correct"""
    functional = case.functional
    functional.weight *= 0
    x = case.rand_x()
    x2 = case.rand_x()
    torch.testing.assert_close(functional(x), functional(x2))  # independent of x if weight is zero
    (prox,) = functional.prox(x, sigma=case.sigma)
    torch.testing.assert_close(prox, x.to(case.result_dtype))  # prox is identity if weight is zero


@pytest.mark.parametrize('functional', FUNCTIONALS)
@pytest.mark.parametrize('case', ('random', 'zero', 'complex'))
def test_functional_grad(functional: type[Functional], case: Literal['random', 'zero', 'complex']):
    """Test if autograd works for functional."""
    rng = RandomGenerator(13)
    match case:
        case 'random':
            x = rng.float32_tensor((1, 2, 3)).requires_grad_(True)
        case 'zero':
            x = torch.zeros(1, requires_grad=True, dtype=torch.float64)
        case 'complex':
            x = rng.complex64_tensor((1, 2, 3)).requires_grad_(True)

    target = rng.float64_tensor((1, 2, 3)).requires_grad_(True)
    weight = rng.float64_tensor((1, 2, 3)).requires_grad_(True)
    f = functional(weight=weight, target=target, dim=(-1, -2))

    with pytest.warns(UserWarning, match='Anomaly Detection has been enabled'):
        with torch.autograd.detect_anomaly():
            x = rng.float64_tensor((1, 2, 3)).requires_grad_(True)
            (fx,) = f(x)
            fx.sum().backward()

    assert x.grad is not None
    torch.autograd.gradcheck(f, x, fast_mode=True)


@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_values(functional: type[ProximableFunctional]):
    """Test if functional values match expected values."""
    # The expected values are taken from the ODL implementation of the functionals.
    # You can use the following code to generate the expected values:

    """
    import odl
    import torch
    torch.manual_seed(42)
    torch.set_printoptions(precision=6)
    functional=odl.solvers.L1Norm # set functional here
    weight,sigma=2,0.5
    space=odl.rn((1,2,3))
    x=torch.arange(-3,3,1).reshape((1,2,3)).to(torch.float32)
    target=torch.randn(1,2,3).round(decimals=2)
    f=(functional(space)*weight).translated(target)
    {'x':x, 'weight':weight, 'target':target, 'sigma':sigma,
    'fx_expected':torch.tensor(f(x)), 'prox_expected:torch.tensor(f.proximal(sigma)(x)),
    'prox_convex_conj_expected':torch.tensor(f.convex_conj.proximal(sigma)(x)))
    }
    """

    cases = {
        'L1Norm': {
            'x': torch.tensor([[[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]]),
            'weight': 2,
            'target': torch.tensor([[[0.340, 0.130, 0.230], [0.230, -1.120, -0.190]]]),
            'sigma': 0.5,
            'fx_expected': torch.tensor(22.480),
            'prox_expected': torch.tensor([[[-2.0, -1.0, 0.0], [0.230, 0.0, 1.0]]]),
            'prox_convex_conj_expected': torch.tensor([[[-2.0, -2.0, -1.115], [-0.115, 1.560, 2.0]]]),
        },
        'L2NormSquared': {
            'x': torch.tensor([[[-3.0, -2.0, -1.0], [0.0, 1.0, 2.0]]]),
            'weight': 2,
            'target': torch.tensor([[[0.340, 0.130, 0.230], [0.230, -1.120, -0.190]]]),
            'sigma': 0.5,
            'fx_expected': torch.tensor(106.195198),
            'prox_expected': torch.tensor([[[-0.328, -0.296, -0.016], [0.184, -0.696, 0.248]]]),
            'prox_convex_conj_expected': torch.tensor(
                [[[-2.983529, -1.943529, -1.049412], [-0.108235, 1.468235, 1.971765]]]
            ),
        },
    }
    if functional.__name__ not in cases:
        pytest.skip(f'No test case for {functional}')
    _test_functional_values(functional, **cases[functional.__name__])


def _test_functional_values(
    functional: type[ProximableFunctional],
    x: torch.Tensor,
    weight: float,
    target: torch.Tensor,
    sigma: float,
    fx_expected: torch.Tensor,
    prox_expected: torch.Tensor,
    prox_convex_conj_expected: torch.Tensor,
):
    f = functional(weight=weight, target=target, dim=None, keepdim=False)
    (fx,) = f(x)
    (prox,) = f.prox(x, sigma=sigma)
    (prox_convex_conj,) = f.prox_convex_conj(x, sigma=sigma)
    torch.testing.assert_close(fx, fx_expected)
    torch.testing.assert_close(prox, prox_expected)
    torch.testing.assert_close(prox_convex_conj, prox_convex_conj_expected)
