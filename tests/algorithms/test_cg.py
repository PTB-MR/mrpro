"""Tests for the conjugate gradient method."""

import pytest
import scipy
import torch
from mrpro.algorithms.optimizers import cg
from mrpro.algorithms.optimizers.cg import CGStatus
from mrpro.operators import EinsumOp, LinearOperatorMatrix
from scipy.sparse.linalg import cg as cg_scp
from tests import RandomGenerator


@pytest.fixture(
    params=[  # (batch-size, vector-size, complex-valued system, separate initial_value)
        (1, 32, False, False),
        (4, 32, True, True),
    ],
    ids=[
        'real_single_noinit',
        'complex_batch',
    ],
)
def system(request):
    """Generate system Hx=b with linear and self-adjoint H."""
    rng = RandomGenerator(seed=0)
    batchsize, vectorsize, complex_valued, separate_initial_value = request.param
    matrix_shape: tuple[int, int, int] = (batchsize, vectorsize, vectorsize)
    vector_shape: tuple[int, int] = (batchsize, vectorsize)
    if complex_valued:
        matrix = rng.complex64_tensor(size=matrix_shape, high=1.0)
        vector = rng.complex64_tensor(size=vector_shape, high=1.0)
    else:
        matrix = rng.float32_tensor(size=matrix_shape, low=-1.0, high=1.0)
        vector = rng.float32_tensor(size=vector_shape, low=-1.0, high=1.0)
    operator = EinsumOp(matrix.mH @ matrix)  # make sure H is self-adjoint
    (right_hand_side,) = operator(vector)
    if separate_initial_value:
        initial_value = rng.rand_like(vector)
    else:
        initial_value = None
    return operator, right_hand_side, vector, initial_value


@pytest.fixture(
    params=[  # (batch-size, vector-size, complex-valued system, separate initial_value)
        (2, 16, (False, True, False), True),
        (3, 16, (False,), False),
    ],
    ids=['3x3-operator-matrix', '1x1-operator-matrix'],
)
def matrixsystem(request):
    """system Hx=b with linear and self-adjoint H as LinearOperatorMatrix."""
    rng = RandomGenerator(seed=1)
    batchsize, vectorsize, complex_valued_system, separate_initial_value = request.param
    matrix_shape: tuple[int, int, int] = (batchsize, vectorsize, vectorsize)
    vector_shape: tuple[int, int] = (batchsize, vectorsize)
    operators = []
    vectors = []
    for complex_operator in complex_valued_system:
        if complex_operator:
            matrix = rng.complex64_tensor(size=matrix_shape, high=1.0)
            vector = rng.complex64_tensor(size=vector_shape, high=1.0)
        else:
            matrix = rng.float32_tensor(size=matrix_shape, low=-1.0, high=1.0)
            vector = rng.float32_tensor(size=vector_shape, low=-1.0, high=1.0)
        vectors.append(vector)
        operators.append(EinsumOp(matrix.mH @ matrix))  # make sure H is self-adjoint
    operator_matrix = LinearOperatorMatrix.from_diagonal(*operators)
    right_hand_side = operator_matrix(*vectors)
    if separate_initial_value:
        initial_value = [rng.rand_like(vector) for vector in vectors]
    else:
        initial_value = None
    return (operator_matrix, right_hand_side, tuple(vectors), initial_value)


def test_cg_solution(system) -> None:
    """Test if CG delivers accurate solution."""
    operator, right_hand_side, solution, initial_value = system
    initial_value = torch.ones_like(solution)
    cg_solution = cg(operator, right_hand_side, initial_value=initial_value, max_iterations=256)
    torch.testing.assert_close(cg_solution, solution, rtol=5e-3, atol=5e-3)


def test_cg_solution_operatormatrix(matrixsystem) -> None:
    """Test if CG delivers accurate solution for a LinearOperatorMatrix."""
    operator, right_hand_side, solution, initial_value = matrixsystem
    cg_solution = cg(
        operator,
        right_hand_side,
        initial_value=initial_value,
        max_iterations=1000,
        tolerance=1e-6,
    )
    torch.testing.assert_close(cg_solution, solution, rtol=5e-3, atol=5e-3)


def test_cg_stopping_after_one_iteration(system) -> None:
    """Test if cg stops after one iteration if the ground-truth is the initial
    guess."""
    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, solution, _ = system

    # callback function; should not be called since cg should exit for loop
    def callback(solution):
        pytest.fail('CG did not exit before performing any iterations')

    # the test should fail if we reach the callback
    xcg_one_iteration = cg(
        operator, right_hand_side, initial_value=solution, max_iterations=10, tolerance=1e-4, callback=callback
    )
    assert (xcg_one_iteration == solution).all()


@pytest.mark.parametrize('max_iterations', [1, 2, 3, 5])
def test_compare_cg_to_scipy(system, max_iterations: int) -> None:
    """Test if our implementation is close to the one of scipy."""
    operator, right_hand_side, _, initial_value = system

    # if batchsize>1, construct H = diag(H1,...,H_batchsize)
    # and b=[b1,...,b_batchsize]^T, otherwise just take the matrix
    matrix_np = scipy.linalg.block_diag(*operator.matrix.numpy())

    (xcg_scipy, _) = cg_scp(
        matrix_np,
        right_hand_side.flatten().numpy(),
        x0=initial_value.flatten().numpy() if initial_value is not None else right_hand_side.flatten().numpy(),
        maxiter=max_iterations,
        atol=0,
    )
    cg_solution_scipy = xcg_scipy.reshape(right_hand_side.shape)
    cg_solution_mrpro = cg(
        operator,
        right_hand_side,
        initial_value=initial_value,
        max_iterations=max_iterations,
        tolerance=0,
    )
    torch.testing.assert_close(cg_solution_mrpro, torch.tensor(cg_solution_scipy), atol=1e-5, rtol=1e-5)


def test_invalid_shapes(system) -> None:
    """Test if CG throws error in case of shape-mismatch."""
    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, *_ = system

    # invalid initial value with mismatched shape
    bad_initial_value = torch.zeros(
        operator.matrix.shape[-1] + 1,
    )
    with pytest.raises(ValueError, match='match'):
        cg(operator, right_hand_side, initial_value=bad_initial_value, max_iterations=10)


def test_callback(system) -> None:
    """Test if the callback function is called if a callback function is set."""
    # create operator, right-hand side
    operator, right_hand_side, _, initial_value = system

    # callback function; if the function is called during the iterations, the
    # test is successful
    def callback(cg_status: CGStatus) -> None:
        _, _, _ = cg_status['iteration_number'], cg_status['solution'][0], cg_status['residual'][0].norm()
        assert True

    cg(operator, right_hand_side, callback=callback)


def test_callback_early_stop(system) -> None:
    operator, right_hand_side, _, initial_value = system
    """Check that when the callback function returns False the optimizer is stopped."""
    callback_check = 0

    # callback function that returns False to stop the algorithm
    def callback(solution):
        nonlocal callback_check
        callback_check += 1
        return False

    cg(operator, right_hand_side, initial_value=initial_value, max_iterations=100, callback=callback)
    assert callback_check == 1


def test_cg_autograd(system) -> None:
    """Test autograd through cg"""
    operator, right_hand_side, _, initial_value = system
    right_hand_side.requires_grad_(True)
    with torch.autograd.detect_anomaly():
        result = cg(operator, right_hand_side, initial_value=initial_value, tolerance=0, max_iterations=5)
        result.abs().sum().backward()
    assert right_hand_side.grad is not None


@pytest.mark.cuda
def test_cg_cuda(matrixsystem) -> None:
    """Test if CG works on CUDA."""
    operator, right_hand_side, solution, initial_value = matrixsystem
    right_hand_side = tuple(x.to('cuda') for x in right_hand_side)
    operator = operator.to('cuda')
    initial_value = tuple(x.to('cuda') for x in initial_value) if initial_value is not None else None
    solution = tuple(x.to('cuda') for x in solution)
    result = cg(operator, right_hand_side, initial_value=initial_value, tolerance=1e-6, max_iterations=1000)
    assert all(x.is_cuda for x in result)
    torch.testing.assert_close(result, solution, rtol=5e-3, atol=5e-3)
