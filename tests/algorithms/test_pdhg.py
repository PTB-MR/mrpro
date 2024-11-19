"""Tests for PDHG."""

import torch
from mrpro.algorithms.optimizers import pdhg
from mrpro.operators import FastFourierOp, IdentityOp, LinearOperatorMatrix, ProximableFunctionalSeparableSum, WaveletOp
from mrpro.operators.functionals import L1Norm, L1NormViewAsReal, L2NormSquared, ZeroFunctional
from tests import RandomGenerator


def test_l2_l1_identification1():
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

    n_iterations = 64
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, n_iterations=n_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_l2_l1_identification2():
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

    n_iterations = 128
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, n_iterations=n_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_fourier_l2_l1_():
    """Set up the problem min_x 1/2*|| Fx - y||_2^2 + lambda * ||x||_1,
    where F is the full FFT sampled on a Cartesian grid. Thus, again, the
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

    n_iterations = 128
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, n_iterations=n_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)


def test_fourier_l2_wavelet_l1_():
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

    n_iterations = 128
    (pdhg_solution,) = pdhg(f=f, g=g, operator=operator, initial_values=initial_values, n_iterations=n_iterations)
    torch.testing.assert_close(pdhg_solution, expected, rtol=5e-4, atol=5e-4)
