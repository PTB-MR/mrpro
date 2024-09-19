import inspect
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Literal, TypedDict

import pytest
import torch
from mrpro.operators import Functional, ProximableFunctional, functionals

from tests import RandomGenerator


@dataclass
class FunctionalTestCase:
    """A testcase for proximable functionals used in test parametrization.

    See functional_test_cases.

    """

    functional: ProximableFunctional
    x_dtype: torch.dtype
    x_shape: torch.Size
    rng: RandomGenerator
    sigma: float | torch.Tensor

    def rand_x(self) -> torch.Tensor:
        """Generate random x"""
        low = 0 if self.x_dtype.is_complex else -2
        return self.rng.rand_tensor(self.x_shape, low=low, high=2.0, dtype=self.x_dtype)

    @property
    def result_dtype(self):
        """Expected result dtype"""
        return torch.promote_types(self.x_dtype, torch.result_type(self.functional.target, self.functional.weight))


FUNCTIONALS: list[type[Functional]] = [
    x[1] for x in inspect.getmembers(functionals, lambda x: inspect.isclass(x) and issubclass(x, Functional))
]
PROXIMABLE_FUNCTIONALS: list[type[ProximableFunctional]] = [
    x for x in FUNCTIONALS if issubclass(x, ProximableFunctional)
]


def functional_test_cases(func: Callable[[FunctionalTestCase], None]) -> None:
    """Decorator combining multiple parameterizations for test cases for all proximable functionals."""

    @pytest.mark.parametrize('shape', [[1, 2, 3]])
    @pytest.mark.parametrize('dtype_name', ['float32', 'complex64'])
    @pytest.mark.parametrize('weight', ['scalar_weight', 'tensor_weight', 'complex_weight'])
    @pytest.mark.parametrize('target', ['no_target', 'random_target'])
    @pytest.mark.parametrize('dim', [None])
    @pytest.mark.parametrize('divide_by_n', [True, False])
    @pytest.mark.parametrize('functional', FUNCTIONALS)
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


@pytest.mark.parametrize('shape', [[1, 2, 3], ()])
@pytest.mark.parametrize('dim', [None, -2])
@pytest.mark.parametrize('keepdim', [True, False])
@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_shape(functional: type[Functional], shape: torch.Size, dim: None | int, keepdim: bool):
    """Test the shape returned by the forward of the functional."""
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


@pytest.mark.parametrize('shape', [[2, 3, 4], ()])
@pytest.mark.parametrize('dim', [None, (0, -2)])
@pytest.mark.parametrize('keepdim', [True, False])
@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_divide_by_n(
    functional: type[Functional], shape: torch.Size, dim: None | tuple[int, ...], keepdim: bool
):
    """Test if divide_by_n scales by number of elements indexed by dim."""

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
    n = torch.sum(ones, dim=dim, keepdim=keepdim)  # elements index by dim
    torch.testing.assert_close(f_mean(x)[0], f_sum(x)[0] / n)


@functional_test_cases
def test_functional_moreau(case: FunctionalTestCase):
    """Test if Moreua identity holds."""
    functional = case.functional
    x = case.rand_x()
    (prox,) = functional.prox(x, sigma=case.sigma)
    (prox_convex_conj,) = functional.prox_convex_conj(x / case.sigma, 1.0 / case.sigma)
    x_new = prox + case.sigma * prox_convex_conj
    torch.testing.assert_close(x.to(case.result_dtype), x_new, rtol=1e-3, atol=1e-3)


@functional_test_cases
def test_functional_prox_optimality(case: FunctionalTestCase):
    """Use autograd to check if prox criterion is minimized."""
    functional = case.functional
    x = case.rand_x()

    (prox,) = functional.prox(x, sigma=case.sigma)

    def prox_criterion(p):
        diff = x - p
        l2 = torch.sum((diff * diff.conj()).real, dim=functional.dim, keepdim=functional.keepdim)
        return (functional(p)[0] + 1 / (2 * case.sigma) * l2).sum()

    for perturbation in (0, 1e-3, 0.1):
        p = (prox + perturbation * case.rng.rand_like(prox)).requires_grad_()
        optim = torch.optim.SGD([p], lr=1e-2)
        for _ in range(200):
            optim.zero_grad()
            prox_criterion(p).backward()
            optim.step()
            optim.param_groups[0]['lr'] *= 0.97

        assert prox_criterion(p) + 1e-5 >= prox_criterion(prox)


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
    # \mathrm{prox}_{\sigma F(\alpha \,\cdot)}(x)=\frac{1}{\alpha}\mathrm{prox}_{\sigma \alpha^2 F(\cdot)}(\alpha x)
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
def test_functional_prox_non_expansive(case: FunctionalTestCase):
    """Tests |function(x) - function(y)|_2 <= |x-y|_2"""
    f = case.functional
    x = case.rand_x()
    y = case.rand_x()

    def l2(x):
        return torch.sum((x * x.conj()).real, dim=case.functional.dim, keepdim=case.functional.keepdim)

    assert (l2(f.prox(x)[0] - f.prox(y)[0]) <= l2(x - y)).all()


@functional_test_cases
def test_functional_prox_zero_weight(case: FunctionalTestCase):
    """Test if scaling with scalar zero weight is correct"""
    functional = case.functional
    functional.weight *= 0
    x = case.rand_x()
    x2 = case.rand_x()
    torch.testing.assert_close(functional(x), functional(x2))  # independent of x if weight is zero
    (prox,) = functional.prox(x, sigma=case.sigma)
    torch.testing.assert_close(prox, x.to(case.result_dtype))  # prox is identity if weight is zero


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_prox_zero_sigma(
    functional: type[ProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test prox with sigma=0"""
    p = getattr(functional(), function)
    x = torch.tensor([-1.0, 0.0, 1.0])
    (prox,) = p(x, sigma=0.0)
    assert not torch.any(torch.isnan(prox))
    assert not torch.any(torch.isinf(prox))
    assert torch.allclose(prox, x)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_negative_sigma(
    functional: type[ProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if error is raised for negative or complex sigma"""

    def case(sigma):
        x = torch.zeros(2)
        f = getattr(functional(keepdim=True), function)
        with pytest.raises(ValueError):
            f(x, sigma)

    # all these cases should throw an exception:
    case(sigma=-1)  # negative
    case(sigma=1j)  # complex
    case(sigma=torch.tensor((1, -1)))  # one element negative
    case(sigma=torch.tensor(1.0, dtype=torch.complex64))  # complex dtype


class NumericCase(TypedDict):
    """The expected values are taken from the ODL implementation of the functionals.
    You can use the following code to generate the expected values:

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

    functional: type[ProximableFunctional]
    x: torch.Tensor
    weight: complex | torch.Tensor
    target: torch.Tensor
    sigma: float
    fx_expected: torch.Tensor
    prox_expected: torch.Tensor
    prox_convex_conj_expected: torch.Tensor


# This is more readable than using pytest.mark.parametrize directly
NUMERICCASES: dict[str, NumericCase] = {  # Name: Case
}


@pytest.mark.parametrize('case_name', NUMERICCASES.keys())
def test_functional_values(case_name: str):
    """Test if functional values match expected values."""
    _test_functional_values(**NUMERICCASES[case_name])


def _test_functional_values(
    functional: type[ProximableFunctional],
    x: torch.Tensor,
    weight: complex | torch.Tensor,
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


@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_has_testcase(functional):
    """Check if there is at least one test case for each functional."""
    cases = [k for k in NUMERICCASES if NUMERICCASES[k]['functional'] == functional]
    assert len(cases), f'No test case found for {functional.__name__}!'


@pytest.mark.cuda()
@pytest.mark.parametrize('functional', FUNCTIONALS)
@pytest.mark.parametrize('parameters', ['scalar', 'none', 'tensor'])
def test_functional_cuda_forward(functional: type[Functional], parameters: Literal['scalar', 'none', 'tensor']):
    """Test if the forward pass works on the GPU."""
    match parameters:
        case 'scalar':
            f = functional(weight=1.0, target=0.0)
        case 'none':
            f = functional()
        case 'tensor':
            f = functional(weight=torch.tensor(1.0), target=torch.tensor(0.0))
    x = torch.tensor(1.0).cuda()
    f.cuda()
    (fx,) = f(x)
    assert fx.is_cuda


@pytest.mark.cuda()
@pytest.mark.parametrize('parameters', ['scalar', 'none', 'tensor'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_cuda_prox(functional: type[ProximableFunctional], parameters: Literal['scalar', 'none', 'tensor']):
    """Test if prox and prox_convex_conj work on the GPU."""
    x = torch.tensor(1.0).cuda()
    match parameters:
        case 'scalar':
            f = functional(weight=1.0, target=0.0).cuda()
            (prox,) = f.prox(x, sigma=1)
            (prox_convex_conj,) = f.prox_convex_conj(x, sigma=1)
        case 'none':
            f = functional().cuda()
            (prox,) = f.prox(x)
            (prox_convex_conj,) = f.prox_convex_conj(x)
        case 'tensor':
            f = functional(weight=torch.tensor(1.0), target=torch.tensor(0.0)).cuda()
            sigma = torch.tensor(1.0).cuda()
            (prox,) = f.prox(x, sigma)
            (prox_convex_conj,) = f.prox_convex_conj(x, sigma)

    assert prox.is_cuda
    assert prox_convex_conj.is_cuda


@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_scalar_arguments_forward(functional: type[Functional]):
    """Test if the forward pass works with scalar weight and target."""
    x = torch.tensor(1.0)
    f_scalar = functional(weight=1.0, target=0.0)
    f_tensor = functional(weight=torch.tensor(1.0), target=torch.tensor(0.0))
    (fx_scalar,) = f_scalar(x)
    (fx_tensor,) = f_tensor(x)
    assert torch.allclose(fx_scalar, fx_tensor)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_scalar_arguments_prox(
    functional: type[ProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if prox and prox_convex_conj work with scalar weight and target."""
    x = torch.tensor(1.0)
    p_scalar = getattr(functional(weight=1.0, target=0.0), function)
    p_tensor = getattr(functional(weight=torch.tensor(1.0), target=torch.tensor(0.0)), function)
    (px_scalar,) = p_scalar(x, sigma=1.0)
    (px_tensor,) = p_tensor(x, sigma=torch.tensor(1.0))
    assert torch.allclose(px_scalar, px_tensor)


@pytest.mark.parametrize('functional', FUNCTIONALS)
def test_functional_default_arguments_forward(functional: type[Functional]):
    """Test if the forward pass works with default weight and target."""
    x = torch.tensor(1.0)
    f_default = functional()
    f_tensor = functional(weight=torch.tensor(1.0), target=torch.tensor(0.0))
    (fx_default,) = f_default(x)
    (fx_tensor,) = f_tensor(x)
    assert torch.allclose(fx_default, fx_tensor)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
@pytest.mark.parametrize('functional', PROXIMABLE_FUNCTIONALS)
def test_functional_default_arguments_prox(
    functional: type[ProximableFunctional], function: Literal['prox', 'prox_convex_conj']
):
    """Test if prox and prox_convex_conj work with default weight and target."""
    x = torch.tensor(1.0)
    p_default = getattr(functional(), function)
    p_tensor = getattr(functional(weight=torch.tensor(1.0), target=torch.tensor(0.0)), function)
    (px_default,) = p_default(x)
    (px_tensor,) = p_tensor(x, sigma=torch.tensor(1.0))
    assert torch.allclose(px_default, px_tensor)
