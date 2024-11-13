"""Tests for the proximal gradient descent."""

import torch
from mrpro.algorithms.optimizers import pgd
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators import FastFourierOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum
from mrpro.operators.functionals import L1Norm, L2NormSquared
from mrpro.phantoms import EllipsePhantom


def test_pgd_convergence_fft_example():
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
    g = L1Norm(divide_by_n=True)

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=0.5,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_convergence_backtracking_fft_example():
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
    g = L1Norm(divide_by_n=True)

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=2.0,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=0.75,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_convergence_backtracking_denoising_example():
    """Test if the proximal gradient descent algorithm converges for denoising."""
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    noise = torch.randn_like(image)
    noisy_image = image + noise

    f = L2NormSquared(target=noisy_image, divide_by_n=True)
    g = L1Norm(divide_by_n=True)

    initial_value = torch.ones_like(image)
    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=2.0,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=0.75,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]


def test_pgd_on_functionals_with_multiple_inputs():
    """Test if the proximal gradient descent algorithm converges for
    the problem min_x f(x) + g(x), with f being a function with multiple inputs,
    g being a ProximableFunctionalSeparableSum."""
    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fft = FastFourierOp()
    (kspace,) = fft(image)

    # to check on multiple inputs, we will use the same input twice
    l2 = L2NormSquared(target=kspace, divide_by_n=True)
    matrix_op = LinearOperatorMatrix(((fft, fft),))
    f = l2 @ matrix_op

    l1 = L1Norm(divide_by_n=True)
    g = ProximableFunctionalSeparableSum(l1, l1)

    initial_value = (torch.ones_like(image), torch.ones_like(image))

    pgd_solution = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=0.5,
        reg_parameter=0.01,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert f(*pgd_solution)[0] + g(*pgd_solution)[0] < f(*initial_value)[0] + g(*initial_value)[0]
