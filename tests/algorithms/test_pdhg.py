"""Tests for PDHG."""

import pytest
import torch
from mrpro.algorithms.optimizers import pdhg
from mrpro.operators import FastFourierOp, IdentityOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum, WaveletOp
from mrpro.operators.functionals import L1Norm, L1NormViewAsReal, L2NormSquared, ZeroFunctional
from tests import RandomGenerator


def test_l2_l1_identification1() -> None:
    """Set up the problem min_x 1/2*||x - y||_2^2 + lambda * ||x||_1,
    which has a closed form solution given by the soft-thresholding operator.

    Here, for f(K(x)) + g(x), we used the identification
        f(x) = 1/2 * || p - y ||_2^2
        g(x) = lambda * ||x||_1
        K = Id
    """
    random_generator = RandomGenerator(seed=0)

    data_shape = (32, 32)
    data = random_generator.float32_tensor(size=data_shape)

    regularization_parameter = 0.1

    l2 = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    l1 = regularization_parameter * L1Norm(divide_by_n=False)

    f = l2
    g = l1
    operator = None  # corresponds to IdentityOp()

    initial_values = (random_generator.float32_tensor(size=data_shape),)
    expected = torch.nn.functional.softshrink(data, regularization_parameter)

    max_iterations = 64
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=max_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_l2_l1_identification2() -> None:
    """Set up the problem min_x 1/2*||x - y||_2^2 + lambda * ||x||_1,
    which has a closed form solution given by the soft-thresholding operator.

    Here, for f(K(x)) + g(x), we used the identification
        f(p,q) = f1(p) + f2(q) = 1/2 * || p - y ||_2^2 + lambda * ||q||_1
        g(x) = 0 for all x,
        K = [Id, Id]^T
    """
    random_generator = RandomGenerator(seed=0)

    data_shape = (32, 64, 64)
    data = random_generator.float32_tensor(size=data_shape)

    regularization_parameter = 0.5

    l2 = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    l1 = regularization_parameter * L1Norm(divide_by_n=False)

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = None  # corresponds to ZeroFunctional()
    operator = LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),)))

    initial_values = (random_generator.float32_tensor(size=data_shape),)

    # solution given by soft thresholding
    expected = torch.nn.functional.softshrink(data, regularization_parameter)

    max_iterations = 128
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=max_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_fourier_l2_l1_() -> None:
    """Set up the problem min_x 1/2*|| Fx - y||_2^2 + lambda * ||x||_1,
    where F is the full FFT and y is sampled on a Cartesian grid. Thus, again, the
    problem has a closed-form solution given by soft-thresholding.
    """
    random_generator = RandomGenerator(seed=0)

    image_shape = (32, 48, 48)
    image = random_generator.complex64_tensor(size=image_shape)

    fourier_op = FastFourierOp(dim=(-3, -2, -1))

    (data,) = fourier_op(image)

    regularization_parameter = 0.5

    l2 = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=False)

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = ZeroFunctional()
    operator = LinearOperatorMatrix(((fourier_op,), (IdentityOp(),)))

    initial_values = (random_generator.complex64_tensor(size=image_shape),)

    # solution given by soft thresholding
    expected = torch.view_as_complex(
        torch.nn.functional.softshrink(torch.view_as_real(fourier_op.H(data)[0]), regularization_parameter)
    )

    max_iterations = 128
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=max_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_fourier_l2_wavelet_l1_() -> None:
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

    (data,) = fourier_op(image)

    regularization_parameter = 0.5

    l2 = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    l1 = regularization_parameter * L1NormViewAsReal(divide_by_n=False)

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = ZeroFunctional()
    operator = LinearOperatorMatrix(((fourier_op,), (wavelet_op,)))

    initial_values = (random_generator.complex64_tensor(size=image_shape),)

    # solution given by soft thresholding
    expected = wavelet_op.H(
        torch.view_as_complex(
            torch.nn.functional.softshrink(
                torch.view_as_real(wavelet_op(fourier_op.H(data)[0])[0]), regularization_parameter
            )
        )
    )[0]

    max_iterations = 128
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=max_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_f_and_g_None() -> None:
    """Check that the initial guess is returned as solution when f and g are None."""
    random_generator = RandomGenerator(seed=0)

    data_shape = (2, 8, 8)

    f = None
    g = None
    operator = None

    initial_values = (random_generator.complex64_tensor(size=data_shape),)

    with pytest.warns(UserWarning, match='constant'):
        (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=1)
    assert (pdhg_solution == initial_values[0]).all()


def test_callback() -> None:
    """Check that the callback function is called."""
    random_generator = RandomGenerator(seed=0)
    f = ZeroFunctional()
    g = None
    operator = None
    initial_values = (random_generator.complex64_tensor(size=(8,)),)

    callback_was_called = False

    # callback function that should be called to change the variable's value to True
    def callback(solution):
        nonlocal callback_was_called
        callback_was_called = True

    pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=1, callback=callback)
    assert callback_was_called


def test_callback_early_stop() -> None:
    """Check that when the callback function returns False the optimizer is stopped."""
    callback_check = 0

    # callback function that returns False to stop the algorithm
    def callback(solution):
        nonlocal callback_check
        callback_check += 1
        return False

    random_generator = RandomGenerator(seed=0)
    f = ZeroFunctional()
    g = None
    operator = None
    initial_values = (random_generator.complex64_tensor(size=(8,)),)

    pdhg(f=f, g=g, operator=operator, initial_values=initial_values, max_iterations=100, callback=callback)
    assert callback_check == 1


def test_stepsizes() -> None:
    """Set up the problem min_x 1/2*||x - y||_2^2 + lambda * ||x||_1,
    which has a closed form solution given by the soft-thresholding operator and check
    that the correct solution is obtained regardless of the chosen stepsizes, i.e.
    when no stepsizes are chosen, within PDHG, the upper-bound of the stepsizes is calculated
    and used, while, if only one of the two is provided, the other stepsize is chosen accordingly.

    Here, for f(K(x)) + g(x), we used the identification
        f(x) = 1/2 * || p - y ||_2^2
        g(x) = lambda * ||x||_1
        K = Id

    """
    random_generator = RandomGenerator(seed=0)

    data_shape = (4, 8, 8)
    data = random_generator.float32_tensor(size=data_shape)

    regularization_parameter = 0.5

    l2 = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    l1 = regularization_parameter * L1Norm(divide_by_n=False)

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = None  # corresponds to ZeroFunctional()
    operator = LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),)))

    # compute the operator norm of the linear operator
    initial_values = (random_generator.float32_tensor(size=data_shape),)
    operator_norm = operator.operator_norm(*initial_values, max_iterations=64)

    expected = torch.nn.functional.softshrink(data, regularization_parameter)

    max_iterations = 256

    # choose primal and dual stepsizes on purpose too high, such that the other one
    # is assigned accordingly within the algorithm to ensure that PDHG converges
    primal_stepsize = 10.0 / operator_norm
    dual_stepsize = 10.0 / operator_norm

    (pdhg_solution_no_step_sizes,) = pdhg(
        f=f,
        g=g,
        operator=operator,
        initial_values=initial_values,
        max_iterations=max_iterations,
    )

    (pdhg_solution_only_primal_stepsize,) = pdhg(
        f=f,
        g=g,
        operator=operator,
        initial_values=initial_values,
        primal_stepsize=primal_stepsize,
        max_iterations=max_iterations,
    )

    (pdhg_solution_only_dual_stepsize,) = pdhg(
        f=f,
        g=g,
        operator=operator,
        initial_values=initial_values,
        dual_stepsize=dual_stepsize,
        max_iterations=max_iterations,
    )

    torch.testing.assert_close(pdhg_solution_no_step_sizes, expected, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(pdhg_solution_only_primal_stepsize, expected, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(pdhg_solution_only_dual_stepsize, expected, rtol=5e-4, atol=5e-4)


def test_value_errors() -> None:
    """Check that value-errors are caught."""
    random_generator = RandomGenerator(seed=0)

    initial_values = (random_generator.complex64_tensor(size=(8,)),)

    with pytest.raises(ValueError, match='same'):
        # len(f) and len(g) are not equal
        pdhg(
            f=ProximableFunctionalSeparableSum(ZeroFunctional(), ZeroFunctional()),
            g=None,
            operator=None,
            initial_values=initial_values,
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='rows'):
        # Number of rows in operator does not match number of functionals in f
        pdhg(
            f=ZeroFunctional(),
            g=None,
            operator=LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),))),
            initial_values=initial_values,
            max_iterations=1,
        )

    with pytest.raises(ValueError, match='columns'):
        # Number of columns in operator does not match number of functionals in f
        pdhg(
            f=None,
            g=ProximableFunctionalSeparableSum(ZeroFunctional(), ZeroFunctional()),
            operator=IdentityOp(),
            initial_values=initial_values,
            max_iterations=1,
        )


def test_pdhg_stopping_after_one_iteration() -> None:
    """Test if pdhg stops after one iteration if the ground-truth is the initial
    guess and the tolerance is high enough."""

    random_generator = RandomGenerator(seed=0)

    data_shape = (1, 2, 3)
    data = random_generator.float32_tensor(size=data_shape)

    regularization_parameter = 2.0

    l2 = 0.5 * L2NormSquared(target=data, divide_by_n=False)
    l1 = regularization_parameter * L1Norm(divide_by_n=False)

    f = ProximableFunctionalSeparableSum(l2, l1)
    g = None  # corresponds to ZeroFunctional()
    operator = LinearOperatorMatrix(((IdentityOp(),), (IdentityOp(),)))

    expected = torch.nn.functional.softshrink(data, regularization_parameter)

    # callback function that should not be called since pdhg should exit the for-loop
    # beforehand
    def callback(solution):
        pytest.fail('PDHG did not exit before performing any iterations')

    max_iterations = 1
    tolerance = 10.0

    pdhg(
        f=f,
        g=g,
        operator=operator,
        tolerance=tolerance,
        initial_values=(expected,),
        max_iterations=max_iterations,
        callback=callback,
    )
