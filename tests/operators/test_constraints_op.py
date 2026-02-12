"""Tests for the Constraints Operator."""

import pytest
import torch
from mr2.operators import ConstraintsOp
from mr2.utils import RandomGenerator

from tests import autodiff_test

BOUNDS = pytest.mark.parametrize(
    'bounds',
    [
        ((None, 1.0),),  # case (-inf, 1.)
        ((1.0, None),),  # case (1, inf)
        ((-5.0, 5.0),),  # case (-5, 5)
        ((-torch.inf, torch.inf),),  # case (-inf, inf)
    ],
    ids=['(-inf, 1.)', '(1, inf)', '(-5, 5)', '(-inf, inf)'],
)


@pytest.mark.parametrize('beta', [1, 0.5, 2])
@BOUNDS
def test_constraints_operator_bounds(bounds: tuple[tuple[float | None, float | None], ...], beta: float) -> None:
    """Tests if the operator correctly applies the bounds."""
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor(size=(36,), low=-100.0, high=100.0)  # large values to be sure to hit bounds
    constraints_op = ConstraintsOp(bounds, beta_sigmoid=beta, beta_softplus=beta)
    (cx,) = constraints_op(x)

    def isset(bound: None | float):
        return bound is not None and torch.tensor(bound).isfinite()

    # check if min/max values of transformed tensor match bounds
    ((lower_bound, upper_bound),) = bounds
    if isset(lower_bound) and isset(upper_bound):  # case (a,b)
        torch.testing.assert_close(cx.min(), torch.tensor(lower_bound))
        torch.testing.assert_close(cx.max(), torch.tensor(upper_bound))
    elif isset(lower_bound):  # case (a, infty)
        torch.testing.assert_close(cx.min(), torch.tensor(lower_bound))
    elif isset(upper_bound):  # case (-infty, b)
        torch.testing.assert_close(cx.max(), torch.tensor(upper_bound))
    else:  # unconstrained case
        torch.testing.assert_close(cx, x)


@BOUNDS
def test_constraints_operator_complex(bounds: tuple[tuple[float | None, float | None], ...], beta: float = 1.0) -> None:
    """Test with complex numbers"""
    # Bounds should be applied to real and imaginary parts separately
    rng = RandomGenerator(seed=0)
    x = rng.complex64_tensor((10,))
    constraints_op = ConstraintsOp(bounds, beta_sigmoid=beta, beta_softplus=beta)

    (actual,) = constraints_op(x)
    expected = constraints_op(x.real)[0] + 1j * constraints_op(x.imag)[0]
    torch.testing.assert_close(actual, expected)

    (inverted,) = constraints_op.inverse(actual)
    torch.testing.assert_close(inverted, x)


@pytest.mark.parametrize('beta', [1, 0.5, 2])
@BOUNDS
def test_constraints_operator_inverse(bounds: tuple[tuple[float | None, float | None], ...], beta: float) -> None:
    """Tests if operator inverse inverses the operator."""
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor(size=(36,))
    constraints_op = ConstraintsOp(bounds, beta_sigmoid=beta, beta_softplus=beta)
    (cx,) = constraints_op(x)
    (xx,) = constraints_op.inverse(cx)
    torch.testing.assert_close(xx, x)


@pytest.mark.parametrize('beta', [1, 0.5, 2])
@BOUNDS
def test_constraints_operator_no_nans(bounds: tuple[tuple[float | None, float | None], ...], beta: float) -> None:
    """Tests if the operator always returns valid values, never nans."""
    rng = RandomGenerator(seed=0)
    x = rng.float32_tensor(size=(36,), low=-100, high=100)
    constraints_op = ConstraintsOp(bounds, beta_sigmoid=beta, beta_softplus=beta)
    (cx,) = constraints_op(x)
    (xx,) = constraints_op.inverse(cx)
    assert not torch.isnan(xx).any()


@pytest.mark.parametrize(
    'bounds',
    [
        ((None, None), (1.0, None), (None, 1.0)),  # matching number of bounds
        ((None, None),),  # fewer bounds than parameters
        ((None, None), (None, None), (None, None), (None, None)),  # more bounds
    ],
    ids=['matching', 'fewer bounds', 'more bounds'],
)
def test_constraints_operator_multiple_inputs(bounds: tuple[tuple[float | None, float | None], ...]) -> None:
    """Tests if the operator works with multiple inputs."""
    rng = RandomGenerator(seed=0)
    # random tensors with arbitrary values
    x1 = rng.float32_tensor(size=(36, 72), low=-1, high=1)
    x2 = rng.float32_tensor(size=(36, 72), low=-1, high=1)
    x3 = rng.float32_tensor(size=(36, 72), low=-1, high=1)

    # define constraints operator using the bounds
    constraints_op = ConstraintsOp(bounds)

    # transform tensor to be component-wise in the range defined by bounds
    cx1, cx2, cx3 = constraints_op(x1, x2, x3)

    # reverse transformation and check if no nans are returned
    xx1, xx2, xx3 = constraints_op.inverse(cx1, cx2, cx3)
    torch.testing.assert_close(xx1, x1)
    torch.testing.assert_close(xx2, x2)
    torch.testing.assert_close(xx3, x3)


@pytest.mark.parametrize(
    'illegal_bounds',
    [
        ((1.0, -1.0), (-1.0, 1.0)),  # first one is invalid
        ((-1.0, 1.0), (1.0, -1.0)),  # second one is invalid
        ((1.0, 1.0),),  # invalid due to 1 == 1
        ((torch.nan, 1),),  # invalid due to first bound being nan
        ((-1, torch.nan),),  # invalid due to second bound being nan
        ((torch.inf, -torch.inf),),  # invalid due to a>b
    ],
)
def test_constraints_operator_illegal_bounds(illegal_bounds: tuple[tuple[float | None, float | None], ...]) -> None:
    """Tests if the operator raises an error for illegal bounds."""
    with pytest.raises(ValueError, match='invalid'):
        ConstraintsOp(illegal_bounds)


def test_autodiff_constraints_operator():
    """Test autodiff works for constraints operator."""
    # random tensors with arbitrary values
    rng = RandomGenerator(seed=0)
    x1 = rng.float32_tensor(size=(36, 72), low=-5, high=5)
    x2 = rng.float32_tensor(size=(36, 72), low=-5, high=5)
    x3 = rng.float32_tensor(size=(36, 72), low=-5, high=5)

    constraints_op = ConstraintsOp(bounds=((None, None), (1.0, None), (None, 1.0)))
    autodiff_test(constraints_op, x1, x2, x3)


@pytest.mark.cuda
def test_constraints_operator_cuda() -> None:
    """Test constraints operator works on CUDA devices."""

    # Generate inputs
    bounds = ((-5.0, 5.0),)
    random_generator = RandomGenerator(seed=0)
    x = random_generator.float32_tensor(size=(36,), low=-100, high=100)

    # Create on CPU, run on CPU
    constraints_op = ConstraintsOp(bounds)
    (cx,) = constraints_op(x)
    assert cx.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    constraints_op = ConstraintsOp(bounds)
    constraints_op.cuda()
    (cx,) = constraints_op(x.cuda())
    assert cx.is_cuda


@pytest.mark.cuda
def test_constraints_operator_inverse_cuda() -> None:
    """Test inverse of constraints operator works on CUDA devices."""

    # Generate inputs
    bounds = ((-5.0, 5.0),)
    random_generator = RandomGenerator(seed=0)
    x = random_generator.float32_tensor(size=(36,), low=-100, high=100)

    # Create on CPU, run on CPU
    constraints_op = ConstraintsOp(bounds)
    (cx,) = constraints_op(x)
    (xx,) = constraints_op.inverse(cx)
    assert xx.is_cpu

    # Create on CPU, transfer to GPU, run on GPU
    constraints_op = ConstraintsOp(bounds)
    constraints_op.cuda()
    (cx,) = constraints_op(x.cuda())
    (xx,) = constraints_op.inverse(cx)
    assert xx.is_cuda
