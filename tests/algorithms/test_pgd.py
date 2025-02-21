"""Tests for the proximal gradient descent."""

import torch
from mrpro.algorithms.optimizers import pgd
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators import FastFourierOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1Norm, L2NormSquared
from mrpro.phantoms import EllipsePhantom


def test_pgd_convergence_fft_example() -> None:
    """Test if the proximal gradient descent algorithm converges for the
    problem min_x 1/2 ||Fx - target||^2 + ||x||_1,
    with F being the Fourier Transform."""
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    l2 = L2NormSquared(target=kspace, divide_by_n=True)
    f = l2 @ fft
    g = 0.01 * L1Norm(divide_by_n=True)

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=0.5,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_convergence_backtracking_fft_example() -> None:
    """Test the backtracking stepsize rule in proximal gradient descent algorithm
    for the problem min_x 1/2 ||Fx - target||^2 + ||x||_1, with F being the
    Fourier Transform."""

    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    l2 = L2NormSquared(target=kspace, divide_by_n=True)
    f = l2 @ fft
    g = 0.01 * L1Norm(divide_by_n=True)

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=2.0,
        max_iterations=200,
        backtrack_factor=0.75,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_convergence_backtracking_denoising_example() -> None:
    """Test if the proximal gradient descent algorithm converges for denoising."""
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    noise = torch.randn_like(image)
    noisy_image = image + noise

    f = L2NormSquared(target=noisy_image, divide_by_n=True)
    g = 0.01 * L1Norm(divide_by_n=True)

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=2.0,
        max_iterations=200,
        backtrack_factor=0.75,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_convergence_on_functionals_with_multiple_inputs() -> None:
    """Test if the proximal gradient descent algorithm converges for
    the problem min_x f(x) + g(x), with f being a function with multiple inputs,
    g being a ProximableFunctionalSeparableSum."""
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    l2 = L2NormSquared(target=kspace, divide_by_n=True)
    l1 = 0.01 * L1Norm(divide_by_n=True)

    # to check on multiple inputs, we will use the same input twice
    f = ProximableFunctionalSeparableSum(l2, l2) @ LinearOperatorMatrix.from_diagonal(fft, fft)
    g = ProximableFunctionalSeparableSum(l1, l1)
    initial_value = (torch.ones_like(image), torch.ones_like(image))

    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=0.5,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_convergence_singular_vs_multiple_inputs() -> None:
    """Test if the proximal gradient descent algorithm converges to the same solution
    in the case of min_x f(x) + g(x) with x single tensor and when f and g are
    separable sum functionals on (x_1,x_2), with x_1 = x_2 = x."""
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    # to check on multiple inputs, we will use the same input twice
    l2 = L2NormSquared(target=kspace, divide_by_n=True)
    l1 = 0.01 * L1Norm(divide_by_n=True)

    pgd_solution_multiple = pgd(
        f=ProximableFunctionalSeparableSum(l2, l2) @ LinearOperatorMatrix.from_diagonal(fft, fft),
        g=ProximableFunctionalSeparableSum(l1, l1),
        initial_value=(torch.ones_like(image), torch.ones_like(image)),
        stepsize=0.5,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    pgd_solution_single = pgd(
        f=l2 @ fft,
        g=l1,
        initial_value=torch.ones_like(image),
        stepsize=0.5,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert torch.allclose(pgd_solution_multiple[0], pgd_solution_single[0])
    assert torch.allclose(pgd_solution_multiple[1], pgd_solution_single[0])
