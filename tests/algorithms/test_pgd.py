"""Tests for the proximal gradient descent."""

import pytest
import torch
from mrpro.algorithms.optimizers import pgd
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.operators import FastFourierOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum, WaveletOp
from mrpro.operators.functionals import L1NormViewAsReal, L2NormSquared
from mrpro.phantoms import EllipsePhantom
from tests import RandomGenerator


@pytest.mark.parametrize(('stepsize', 'backtrack_factor'), [(1, 1), (1e5, 0.8)])
def test_pgd_solution_fourier_l1(stepsize, backtrack_factor) -> None:
    """ "Set up the problem min_x 1/2*|| Fx - y||_2^2 + lambda * ||x||_1,
    where F is the full FFT and y is sampled on a Cartesian grid. Thus the
    problem has a closed-form solution given by soft-thresholding. Test
    if the expected solution and the one obtained by the pgd are close."""

    random_generator = RandomGenerator(seed=0)
    image_shape = (6, 32, 32)
    image = random_generator.complex64_tensor(size=image_shape)
    dim = (-3, -2, -1)

    fourier_op = FastFourierOp(dim=dim)
    (kspace,) = fourier_op(image)

    l2 = 0.5 * L2NormSquared(target=kspace, divide_by_n=False)
    f = l2 @ fourier_op

    regularization_parameter = 0.1
    g = regularization_parameter * L1NormViewAsReal(divide_by_n=False)

    initial_value = torch.ones_like(image)

    # solution given by soft thresholding
    expected = torch.view_as_complex(
        torch.nn.functional.softshrink(torch.view_as_real(fourier_op.H(kspace)[0]), regularization_parameter)
    )

    max_iterations = 150
    (pgd_solution,) = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=stepsize,
        max_iterations=max_iterations,
        backtrack_factor=backtrack_factor,
        convergent_iterates_variant=False,
    )
    torch.testing.assert_close(pgd_solution, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize(('stepsize', 'backtrack_factor'), [(1, 1), (1e5, 0.8)])
def test_pgd_solution_fourier_wavelet(stepsize, backtrack_factor) -> None:
    """Set up the problem min_x 1/2*|| Fx - y||_2^2 + lambda * || W x||_1,
    where F is the full FFT sampled on a Cartesian grid and W a wavelet transform.
    Because both F and W are invertible and preserve the norm, the problem has a closed-form solution
    obtainable by soft-thresholding.
    """
    random_generator = RandomGenerator(seed=0)
    image_shape = (6, 32, 32)
    image = random_generator.complex64_tensor(size=image_shape)
    dim = (-3, -2, -1)

    fourier_op = FastFourierOp(dim=dim)
    wavelet_op = WaveletOp(domain_shape=image_shape, dim=dim)

    (kspace,) = fourier_op(image)

    l2 = 0.5 * L2NormSquared(target=kspace, divide_by_n=False)
    f = l2 @ fourier_op @ wavelet_op.H

    regularization_parameter = 0.1
    g = regularization_parameter * L1NormViewAsReal(divide_by_n=False)

    # solution given by soft thresholding and back to the image space
    expected = wavelet_op.H(
        torch.view_as_complex(
            torch.nn.functional.softshrink(
                torch.view_as_real(wavelet_op(fourier_op.H(kspace)[0])[0]), regularization_parameter
            )
        )
    )[0]

    max_iterations = 150
    initial_value = torch.ones_like(wavelet_op(image)[0])

    (pgd_solution_wave,) = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=stepsize,
        max_iterations=max_iterations,
        backtrack_factor=backtrack_factor,
    )

    (pgd_solution_image,) = wavelet_op.H(pgd_solution_wave)

    torch.testing.assert_close(pgd_solution_image, expected, rtol=5e-4, atol=5e-4)


def test_callback() -> None:
    """Check that the callback function is called."""
    random_generator = RandomGenerator(seed=0)
    f = L2NormSquared(target=torch.zeros((1, 10, 10)), divide_by_n=False)
    g = L1NormViewAsReal(divide_by_n=False)
    initial_values = (random_generator.complex64_tensor(size=(1, 10, 10)),)

    callback_was_called = False

    # callback function that should be called to change the variable's value to True
    def callback(solution):
        nonlocal callback_was_called
        callback_was_called = True

    pgd(f=f, g=g, initial_value=initial_values, max_iterations=1, callback=callback)
    assert callback_was_called


def test_pgd_behavior_singular_vs_multiple_inputs() -> None:
    """Test if the proximal gradient descent algorithm returns the same solution
    in the case of min_x f(x) + g(x) with x single tensor and when f and g are
    separable sum functionals on (x_1,x_2), with x_1 = x_2 = x."""

    dim = SpatialDimension(x=100, y=100, z=1)
    ellipse = EllipsePhantom()
    image = ellipse.image_space(dim)
    fourier_op = FastFourierOp()
    (kspace,) = fourier_op(image)

    l2 = L2NormSquared(target=kspace, divide_by_n=True)
    l1 = 0.01 * L1NormViewAsReal(divide_by_n=True)

    # to check on multiple inputs, we will use the same input twice
    pgd_solution_multiple = pgd(
        f=ProximableFunctionalSeparableSum(l2, l2) @ LinearOperatorMatrix.from_diagonal(fourier_op, fourier_op),
        g=ProximableFunctionalSeparableSum(l1, l1),
        initial_value=(torch.ones_like(image), torch.ones_like(image)),
        stepsize=0.5,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    pgd_solution_single = pgd(
        f=l2 @ fourier_op,
        g=l1,
        initial_value=torch.ones_like(image),
        stepsize=0.5,
        max_iterations=200,
        backtrack_factor=1.0,
    )

    assert torch.allclose(pgd_solution_multiple[0], pgd_solution_single[0])
    assert torch.allclose(pgd_solution_multiple[1], pgd_solution_single[0])


def test_pgd_behavior_different_updates_t() -> None:
    """Test if the proximal gradient descent algorithm returns the same solution
    for the two different updates of t."""

    random_generator = RandomGenerator(seed=0)
    image_shape = (6, 32, 32)
    image = random_generator.complex64_tensor(size=image_shape)
    dim = (-3, -2, -1)
    fourier_op = FastFourierOp(dim=dim)
    (kspace,) = fourier_op(image)

    l2 = 0.5 * L2NormSquared(target=kspace, divide_by_n=False)
    f = l2 @ fourier_op

    regularization_parameter = 0.1
    g = regularization_parameter * L1NormViewAsReal(divide_by_n=False)

    initial_value = torch.ones_like(image)

    max_iterations = 100

    (pgd_solution1,) = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=1.0,
        max_iterations=max_iterations,
        backtrack_factor=0.9,
        convergent_iterates_variant=False,
    )

    (pgd_solution2,) = pgd(
        f=f,
        g=g,
        initial_value=initial_value,
        stepsize=1.0,
        max_iterations=max_iterations,
        backtrack_factor=0.9,
        convergent_iterates_variant=True,
    )
    torch.testing.assert_close(pgd_solution1, pgd_solution2, rtol=5e-4, atol=5e-4)


# %%
