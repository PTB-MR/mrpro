"""Tests the conjugate gradient operator."""

import torch
from mr2.operators import ConjugateGradientOp, EinsumOp, LinearOperatorMatrix
from mr2.utils import RandomGenerator


def random_linearop(size: tuple[int, int], rng: RandomGenerator):
    """Create a random LinearOperator."""
    return EinsumOp(rng.complex128_tensor(size), '... i j, ... j -> ... i')


def test_conjugate_gradient_op_least_squares_matrix(sizes: tuple[int, int] = (10, 8), noise_level: float = 1e-3):
    """Test the conjugate gradient operator for |Ax-y|^2+alpha*|x-x0|^2 with a Matrix of LinearOperators."""
    rng = RandomGenerator(0)
    a = LinearOperatorMatrix.from_diagonal(*(random_linearop((s, s), rng) for s in sizes))
    x = tuple(rng.complex128_tensor((s,)) for s in sizes)
    y = a(*x)
    y = tuple((yi + rng.rand_like(yi) * noise_level).requires_grad_(True) for yi in y)
    x0 = tuple((xi + rng.rand_like(xi) * noise_level).requires_grad_(True) for xi in x)
    n_y = len(sizes)
    op = ConjugateGradientOp(
        lambda alpha, *_: a.gram + alpha,
        lambda alpha, *y_x0: tuple(ahy + alpha * x0 for ahy, x0 in zip(a.H(*y_x0[:n_y]), y_x0[n_y:], strict=True)),
    )
    alpha = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    solution = op(alpha, *y, *x0)
    loss = sum((((si - xi).norm()) for si, xi in zip(solution, x, strict=True)), torch.tensor(0.0))
    assert loss.item() < 0.01
    loss.backward()
    assert alpha.grad is not None
    for x0i in x0:
        assert x0i.grad is not None
    for yi in y:
        assert yi.grad is not None


def test_conjugate_gradient_op_least_squares_gradcheck_unrolled(size: int = 10, noise_level: float = 1e-2):
    """Test the implicit differentiation of the conjugate gradient operator using |Ax-y|^2+alpha*|x-x0|^2."""
    rng = RandomGenerator(0)
    a = random_linearop((size, size), rng)
    x = rng.complex128_tensor((size,))
    (y,) = a(x)
    y = y + rng.rand_like(y) * noise_level
    y.requires_grad = True
    x0 = x + rng.rand_like(x) * noise_level
    x0.requires_grad = True
    op = ConjugateGradientOp(
        lambda alpha, _x0, _y: a.gram + alpha,
        lambda alpha, x0, y: (a.H(y)[0] + alpha * x0,),
        implicit_backward=False,
    )
    alpha = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(op, (alpha, y, x0), fast_mode=True)


def test_conjugate_gradient_op_least_squares_gradcheck_implicit(size: int = 10, noise_level: float = 1e-2):
    """Test the implicit differentiation of the conjugate gradient operator using |Ax-y|^2+alpha*|x-x0|^2."""
    rng = RandomGenerator(0)
    a = random_linearop((size, size), rng)
    x = rng.complex128_tensor((size,))
    (y,) = a(x)
    y = y + rng.rand_like(y) * noise_level
    y.requires_grad = True
    x0 = x + rng.rand_like(x) * noise_level
    x0.requires_grad = True
    op = ConjugateGradientOp(
        lambda alpha, _x0, _y: a.gram + alpha,
        lambda alpha, x0, y: (a.H(y)[0] + alpha * x0,),
    )
    alpha = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(op, (alpha, y, x0), fast_mode=True)
