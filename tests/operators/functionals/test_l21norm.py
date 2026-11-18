"""Tests for the L21Norm (adapted from mrpro tests/operators/functionals/test_functional.py + conftest.py)."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from mrpro.operators.functionals import L21Norm  # or: from mrpro.operators.functionals import L21Norm
from mrpro.utils import RandomGenerator
from mrpro.utils.reshape import normalize_indices

# ----------------------------------------------------------------------------------------------
# Parametrization shared by all tests
# ----------------------------------------------------------------------------------------------
SHAPES = [
    [2, 2, 3, 4],
]
DIMS = [-1, -2]
KERNEL_DIMS = [0, 1]
KEEPDIMS = [False, True]


# ----------------------------------------------------------------------------------------------
# "conftest" part: test case + decorator (the L21Norm needs kernel_dim and a weight that is
# constant along kernel_dim, therefore we cannot reuse the generic conftest)
# ----------------------------------------------------------------------------------------------
@dataclass
class L21TestCase:
    """A testcase for the L21Norm used in test parametrization."""

    functional: L21Norm
    x_dtype: torch.dtype
    x_shape: torch.Size
    rng: RandomGenerator
    sigma: float | torch.Tensor
    kernel_dim: int
    dim: int

    def rand_x(self) -> torch.Tensor:
        """Generate random x."""
        low = 0 if self.x_dtype.is_complex else -2
        return self.rng.rand_tensor(self.x_shape, low=low, high=2.0, dtype=self.x_dtype)

    @property
    def group_shape(self) -> tuple[int, ...]:
        """Shape that is constant (=1) along kernel_dim, full size elsewhere (valid shape for weight/alpha)."""
        shape = list(self.x_shape)
        shape[self.kernel_dim] = 1
        return tuple(shape)

    @property
    def result_dtype(self):
        """Expected result dtype."""
        return torch.promote_types(self.x_dtype, torch.result_type(self.functional.target, self.functional.weight))


def l21_test_cases(func: Callable[[L21TestCase], None]) -> Callable[..., None]:
    """Decorator combining multiple parameterizations of test cases for the L21Norm."""

    @pytest.mark.parametrize('shape', SHAPES)
    @pytest.mark.parametrize('dim', DIMS)
    @pytest.mark.parametrize('kernel_dim', KERNEL_DIMS)
    @pytest.mark.parametrize('keepdim', KEEPDIMS)
    @pytest.mark.parametrize('dtype_name', ['float32', 'complex64'])
    @pytest.mark.parametrize('weight', ['scalar_weight', 'tensor_weight', 'binary_weight'])
    @pytest.mark.parametrize('target', ['no_target', 'random_target'])
    @pytest.mark.parametrize('divide_by_n', [True, False], ids=['mean', 'sum'])
    def wrapper(
        shape: torch.Size,
        dim: int,
        kernel_dim: int,
        keepdim: bool,
        dtype_name: Literal['float32', 'complex64'],
        weight: Literal['scalar_weight', 'tensor_weight', 'binary_weight'],
        target: Literal['no_target', 'random_target'],
        divide_by_n: bool,
    ):
        dtype = getattr(torch, dtype_name)
        rng = RandomGenerator(13)
        sigma = rng.rand_tensor((1,), low=0.0, high=1.0, dtype=dtype.to_real())

        # The weight has to be constant along kernel_dim -> size 1 along kernel_dim.
        # The implementation uses sqrt(clamp_min(weight, 0)), so we use non-negative weights.
        weight_shape = list(shape)
        weight_shape[kernel_dim] = 1
        match weight:
            case 'scalar_weight':
                weight_value: float | torch.Tensor = rng.float64(low=0.1, high=2)
            case 'tensor_weight':
                weight_value = rng.rand_tensor(tuple(weight_shape), low=0.1, high=1.0, dtype=dtype.to_real())
            case 'binary_weight':
                weight_value = rng.rand_tensor(tuple(weight_shape), low=0.0, high=1.0, dtype=dtype.to_real()).round()
        match target:
            case 'no_target':
                target_value: torch.Tensor | None = None
            case 'random_target':
                target_value = rng.rand_tensor(shape, low=0 if dtype.is_complex else -1, high=1.0, dtype=dtype)

        f = L21Norm(
            divide_by_n=divide_by_n,
            weight=weight_value,
            target=target_value,
            dim=dim,
            kernel_dim=kernel_dim,
            keepdim=keepdim,
        )
        case = L21TestCase(
            functional=f, x_shape=shape, x_dtype=dtype, rng=rng, sigma=sigma, kernel_dim=kernel_dim, dim=dim
        )
        func(case)

    return wrapper


# ----------------------------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------------------------
@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('keepdim', KEEPDIMS)
@pytest.mark.parametrize('kernel_dim', KERNEL_DIMS)
def test_functional_shape(shape: torch.Size, dim: int, keepdim: bool, kernel_dim: int):
    """Test the shape returned by the forward of the functional."""
    f = L21Norm(dim=dim, kernel_dim=kernel_dim, keepdim=keepdim)
    x = torch.ones(shape)

    if dim is not None and (dim >= len(shape) or dim < -len(shape)):
        # dim is out of bounds
        # easier to check here than to remove the cases from the parametrization
        with pytest.raises(IndexError):
            f(x)
        return

    (fx,) = f(x)

    # easier to calculate the expected shape than to parametrize it
    combined_dim = (kernel_dim, dim)

    if not keepdim:
        normalized = normalize_indices(len(shape), combined_dim)
        expected_shape = tuple(s for i, s in enumerate(shape) if i not in normalized)
    elif keepdim:
        normalized = normalize_indices(len(shape), combined_dim)
        expected_shape = tuple(s if i not in normalized else 1 for i, s in enumerate(shape))
    assert fx.shape == expected_shape


@pytest.mark.parametrize('shape', SHAPES)
@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('keepdim', KEEPDIMS)
@pytest.mark.parametrize('kernel_dim', KERNEL_DIMS)
def test_functional_divide_by_n(shape: torch.Size, dim: int, keepdim: bool, kernel_dim: int):
    """Test if divide_by_n scales by number of elements indexed by dim."""

    f_mean = L21Norm(divide_by_n=True, dim=dim, keepdim=keepdim, kernel_dim=kernel_dim)
    f_sum = L21Norm(divide_by_n=False, dim=dim, keepdim=keepdim, kernel_dim=kernel_dim)
    x = RandomGenerator(13).float32_tensor(shape)

    dim_combined = (kernel_dim, dim)
    if dim_combined is not None and any(d >= len(shape) or d < -len(shape) for d in dim_combined):
        # dim is out of bounds
        # easier to check here than to remove the cases from the parametrization
        with pytest.raises(IndexError):
            f_mean(x)
        with pytest.raises(IndexError):
            f_sum(x)
        return

    n = x.shape[dim]  # elements index by dim
    torch.testing.assert_close(f_mean(x)[0], f_sum(x)[0] / n)


@l21_test_cases
def test_functional_moreau(case: L21TestCase):
    """Test if Moreau identity holds."""
    functional = case.functional
    x = case.rand_x()
    (prox,) = functional.prox(x, sigma=case.sigma)
    (prox_convex_conj,) = functional.prox_convex_conj(x / case.sigma, 1.0 / case.sigma)
    x_new = prox + case.sigma * prox_convex_conj
    torch.testing.assert_close(x.to(case.result_dtype), x_new, rtol=1e-3, atol=1e-3)


@l21_test_cases
def test_functional_prox_optimality(case: L21TestCase):
    """Use autograd to check if prox criterion is minimized."""
    functional = case.functional
    x = case.rand_x()

    (prox,) = functional.prox(x, sigma=case.sigma)

    def prox_criterion(p: torch.Tensor) -> torch.Tensor:
        diff = x - p
        # the prox criterion is separable over the batch entries -> summing everything is fine
        l2 = torch.sum((diff * diff.conj()).real)
        return (case.sigma * functional(p)[0]).sum() + 1 / 2 * l2

    for perturbation in (0, 1e-3, 0.1):
        p = (prox + perturbation * case.rng.rand_like(prox)).requires_grad_()
        optim = torch.optim.SGD([p], lr=1e-2)
        for _ in range(200):
            optim.zero_grad()
            prox_criterion(p).backward()
            optim.step()
            optim.param_groups[0]['lr'] *= 0.97

        reference = prox_criterion(prox)
        # relative tolerance, since the shapes (and thus the criterion values) are larger than in the generic test
        assert prox_criterion(p) + 1e-5 + 1e-5 * reference.abs() >= reference


@l21_test_cases
def test_functional_shift(case: L21TestCase):
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


@l21_test_cases
def test_functional_scaling(case: L21TestCase):
    """Test if scaling with a (per group) scalar factor is correct.

    The L21Norm is f(x) = sum_i sqrt(w_i) ||(x-b)_i||_2, i.e. the weight enters as sqrt(w).
    With G(x) = F(alpha x) we therefore get G = L21Norm(weight=w*alpha^2, target=b/alpha):
        sqrt(w) * alpha * ||x - b/alpha|| = sqrt(w) * ||alpha x - b||
    and
        prox_{sigma F(alpha .)}(x) = 1/alpha * prox_{sigma alpha^2 F}(alpha x)
    alpha has to be constant along kernel_dim (otherwise the weight would vary along kernel_dim).
    """
    alpha = case.rng.rand_tensor(case.group_shape, low=0.5, high=2.0, dtype=torch.float32)
    functional = case.functional
    x = case.rand_x()
    fx_target = functional(alpha * x)[0]
    prox_target = 1 / alpha * functional.prox(alpha * x, sigma=alpha**2 * case.sigma)[0]
    scaled_functional = deepcopy(functional)
    scaled_functional.weight = functional.weight * alpha**2
    scaled_functional.target = functional.target / alpha
    fx_scaled = scaled_functional(x)[0]
    prox_scaled = scaled_functional.prox(x, sigma=case.sigma)[0]
    torch.testing.assert_close(prox_scaled, prox_target)
    torch.testing.assert_close(fx_scaled, fx_target)


@l21_test_cases
def test_functional_prox_non_expansive(case: L21TestCase):
    """Tests |prox(x) - prox(y)|_2 <= |x-y|_2 (for each group along kernel_dim)."""
    f = case.functional
    x = case.rand_x()
    y = case.rand_x()

    def l2(v):
        return torch.sum((v * v.conj()).real, dim=case.kernel_dim, keepdim=True)

    lhs = l2(f.prox(x)[0] - f.prox(y)[0])
    rhs = l2(x - y)
    # groups with weight 0 give prox = identity -> equality up to float32 rounding
    assert (lhs <= rhs * (1 + 1e-4) + 1e-5).all()


@l21_test_cases
def test_functional_prox_zero_weight(case: L21TestCase):
    """Test if scaling with scalar zero weight is correct."""
    functional = case.functional
    functional.weight = functional.weight * 0
    x = case.rand_x()
    x2 = case.rand_x()
    torch.testing.assert_close(functional(x)[0], functional(x2)[0])  # independent of x if weight is zero
    (prox,) = functional.prox(x, sigma=case.sigma)
    torch.testing.assert_close(prox, x.to(case.result_dtype))  # prox is identity if weight is zero


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('kernel_dim', KERNEL_DIMS)
@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
def test_functional_prox_zero_sigma(dim: int, kernel_dim: int, function: Literal['prox', 'prox_convex_conj']):
    """Test prox with sigma=0."""
    p = getattr(L21Norm(dim=dim, kernel_dim=kernel_dim), function)
    # small values, such that every group has norm < 1 (the prox_convex_conj projects onto the unit ball)
    x = 0.1 * RandomGenerator(13).float32_tensor((2, 3, 4, 5), low=-1.0, high=1.0)
    (prox,) = p(x, sigma=0.0)
    assert not torch.any(torch.isnan(prox))
    assert not torch.any(torch.isinf(prox))
    assert torch.allclose(prox, x)


@pytest.mark.parametrize('dim', DIMS)
@pytest.mark.parametrize('kernel_dim', KERNEL_DIMS)
@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
def test_functional_negative_sigma(dim: int, kernel_dim: int, function: Literal['prox', 'prox_convex_conj']):
    """Test if error is raised for negative or complex sigma."""

    def case(sigma):
        x = torch.zeros(2, 3, 4, 5)
        f = getattr(L21Norm(dim=dim, kernel_dim=kernel_dim, keepdim=True), function)
        with pytest.raises(ValueError):
            f(x, sigma)

    # all these cases should throw an exception:
    case(sigma=-1)  # negative
    case(sigma=1j)  # complex
    case(sigma=torch.tensor((1, -1)))  # one element negative
    case(sigma=torch.tensor(1.0, dtype=torch.complex64))  # complex dtype


# ----------------------------------------------------------------------------------------------
# Numeric test with hand-computed values
# ----------------------------------------------------------------------------------------------
# x has shape (2, 3); kernel_dim=0 -> the groups are the columns (3,4), (0,0), (1,0) with L2 norms 5, 0, 1.
# dim=1 -> outer sum (or mean) over the three columns. weight=4 -> sqrt(weight)=2, sigma=0.5, no target.
#
# sum:  f      = 2 * (5 + 0 + 1) = 12
#       prox   : threshold alpha = 2 * 0.5 = 1      -> scales (1-1/5, -, 0)  = (0.8, -, 0)
#       p.c.c. : projection on ball with radius 2   -> scales (2/5, -, 1)
# mean: f      = 12 / 3 = 4
#       prox   : threshold alpha = 1 / 3            -> scales (1-1/15, -, 1-1/3)
#       p.c.c. : projection on ball with radius 2/3 -> scales (2/15, -, 2/3)
NUMERIC_X = torch.tensor([[3.0, 0.0, 1.0], [4.0, 0.0, 0.0]])
NUMERICCASES = {
    'sum': {
        'divide_by_n': False,
        'fx_expected': torch.tensor(12.0),
        'prox_expected': torch.tensor([[2.4, 0.0, 0.0], [3.2, 0.0, 0.0]]),
        'prox_convex_conj_expected': torch.tensor([[1.2, 0.0, 1.0], [1.6, 0.0, 0.0]]),
    },
    'mean': {
        'divide_by_n': True,
        'fx_expected': torch.tensor(4.0),
        'prox_expected': torch.tensor([[3 * 14 / 15, 0.0, 2 / 3], [4 * 14 / 15, 0.0, 0.0]]),
        'prox_convex_conj_expected': torch.tensor([[3 * 2 / 15, 0.0, 2 / 3], [4 * 2 / 15, 0.0, 0.0]]),
    },
}


@pytest.mark.parametrize('case_name', NUMERICCASES.keys())
def test_functional_values(case_name: str):
    """Test if functional values match the hand-computed expected values."""
    c = NUMERICCASES[case_name]
    f = L21Norm(weight=4.0, target=None, dim=1, kernel_dim=0, keepdim=False, divide_by_n=bool(c['divide_by_n']))
    (fx,) = f(NUMERIC_X)
    (prox,) = f.prox(NUMERIC_X, sigma=0.5)
    (prox_convex_conj,) = f.prox_convex_conj(NUMERIC_X, sigma=0.5)
    torch.testing.assert_close(fx, c['fx_expected'])
    torch.testing.assert_close(prox, c['prox_expected'])
    torch.testing.assert_close(prox_convex_conj, c['prox_convex_conj_expected'])


# ----------------------------------------------------------------------------------------------
# L21-specific: argument validation
# ----------------------------------------------------------------------------------------------
def test_missing_kernel_dim():
    """kernel_dim is required."""
    with pytest.raises(ValueError):
        L21Norm(dim=1)


def test_overlapping_dim_and_kernel_dim():
    """dim and kernel_dim must be disjoint."""
    with pytest.raises(ValueError):
        L21Norm(dim=(0, 1), kernel_dim=1)


@pytest.mark.parametrize('function', ['__call__', 'prox', 'prox_convex_conj'])
def test_weight_varying_along_kernel_dim_raises(function: str):
    """A weight that varies along kernel_dim is not supported and has to raise."""
    f = L21Norm(weight=torch.rand(2, 3) + 0.5, dim=1, kernel_dim=0)
    with pytest.raises(ValueError):
        getattr(f, function)(torch.ones(2, 3))


def test_weight_constant_along_kernel_dim_is_accepted():
    """Weight with size 1 along kernel_dim (but varying along dim) works."""
    f = L21Norm(weight=torch.rand(1, 3) + 0.5, dim=1, kernel_dim=0)
    x = torch.ones(2, 3)
    f(x)
    f.prox(x)
    f.prox_convex_conj(x)


# ----------------------------------------------------------------------------------------------
# CUDA / scalar / default argument tests
# ----------------------------------------------------------------------------------------------
def _make(parameters: Literal['scalar', 'none', 'tensor']) -> L21Norm:
    match parameters:
        case 'scalar':
            return L21Norm(weight=1.0, target=0.0, dim=1, kernel_dim=0)
        case 'none':
            return L21Norm(dim=1, kernel_dim=0)
        case 'tensor':
            return L21Norm(weight=torch.tensor(1.0), target=torch.tensor(0.0), dim=1, kernel_dim=0)


@pytest.mark.cuda
@pytest.mark.parametrize('parameters', ['scalar', 'none', 'tensor'])
def test_functional_cuda_forward(parameters: Literal['scalar', 'none', 'tensor']):
    """Test if the forward pass works on the GPU."""
    f = _make(parameters)
    x = torch.ones(2, 3).cuda()
    f.cuda()
    (fx,) = f(x)
    assert fx.is_cuda


@pytest.mark.cuda
@pytest.mark.parametrize('parameters', ['scalar', 'none', 'tensor'])
def test_functional_cuda_prox(parameters: Literal['scalar', 'none', 'tensor']):
    """Test if prox and prox_convex_conj work on the GPU."""
    x = torch.ones(2, 3).cuda()
    f = _make(parameters).cuda()
    if parameters == 'tensor':
        sigma: float | torch.Tensor = torch.tensor(1.0).cuda()
        (prox,) = f.prox(x, sigma)
        (prox_convex_conj,) = f.prox_convex_conj(x, sigma)
    elif parameters == 'scalar':
        (prox,) = f.prox(x, sigma=1)
        (prox_convex_conj,) = f.prox_convex_conj(x, sigma=1)
    else:
        (prox,) = f.prox(x)
        (prox_convex_conj,) = f.prox_convex_conj(x)
    assert prox.is_cuda
    assert prox_convex_conj.is_cuda


def test_functional_scalar_arguments_forward():
    """Test if the forward pass works with scalar weight and target."""
    x = torch.ones(2, 3)
    (fx_scalar,) = _make('scalar')(x)
    (fx_tensor,) = _make('tensor')(x)
    assert torch.allclose(fx_scalar, fx_tensor)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
def test_functional_scalar_arguments_prox(function: Literal['prox', 'prox_convex_conj']):
    """Test if prox and prox_convex_conj work with scalar weight and target."""
    x = torch.ones(2, 3)
    p_scalar = getattr(_make('scalar'), function)
    p_tensor = getattr(_make('tensor'), function)
    (px_scalar,) = p_scalar(x, sigma=1.0)
    (px_tensor,) = p_tensor(x, sigma=torch.tensor(1.0))
    assert torch.allclose(px_scalar, px_tensor)


def test_functional_default_arguments_forward():
    """Test if the forward pass works with default weight and target."""
    x = torch.ones(2, 3)
    (fx_default,) = _make('none')(x)
    (fx_tensor,) = _make('tensor')(x)
    assert torch.allclose(fx_default, fx_tensor)


@pytest.mark.parametrize('function', ['prox', 'prox_convex_conj'])
def test_functional_default_arguments_prox(function: Literal['prox', 'prox_convex_conj']):
    """Test if prox and prox_convex_conj work with default weight and target."""
    x = torch.ones(2, 3)
    p_default = getattr(_make('none'), function)
    p_tensor = getattr(_make('tensor'), function)
    (px_default,) = p_default(x)
    (px_tensor,) = p_tensor(x, sigma=torch.tensor(1.0))
    assert torch.allclose(px_default, px_tensor)
