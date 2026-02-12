"""Tests for the conjugate gradient method."""

import pytest
import scipy.linalg
import scipy.sparse
import torch
from mr2.algorithms.optimizers import cg
from mr2.algorithms.optimizers.cg import CGStatus
from mr2.operators import EinsumOp, LinearOperatorMatrix
from mr2.utils import RandomGenerator


@pytest.fixture(
    params=[  # (batch-size, vector-size, complex-valued system, separate initial_value)
        ((), 32, False, False),
        ((4,), 32, True, True),
    ],
    ids=[
        'real_single_noinit',
        'complex_batch',
    ],
)
def system(request):
    """Generate system Hx=b with linear and self-adjoint H."""
    rng = RandomGenerator(seed=123)
    batchsize, vectorsize, complex_valued, separate_initial_value = request.param
    matrix_shape: tuple[int, int, int] = (*batchsize, vectorsize, vectorsize)
    vector_shape: tuple[int, int] = (*batchsize, vectorsize)
    if complex_valued:
        matrix = rng.complex64_tensor(size=matrix_shape, high=1.0)
    else:
        matrix = rng.float64_tensor(size=matrix_shape, low=-1.0, high=1.0)

    # make sure H is self-adjoint
    self_adjoint_matrix = matrix.mH @ matrix

    # construct matrix multiplication as LinearOperator
    operator = EinsumOp(self_adjoint_matrix)

    # create ground-truth data and right-hand side of the system
    if complex_valued:
        vector = rng.complex64_tensor(size=vector_shape, high=1.0)
    else:
        vector = rng.float64_tensor(size=vector_shape, low=-1.0, high=1.0)

    (right_hand_side,) = operator(vector)
    if separate_initial_value:
        initial_value = rng.rand_like(vector)
    else:
        initial_value = None
    return operator, right_hand_side, vector, initial_value


@pytest.fixture(
    params=[  # (batch-size, vector-size, complex-valued system, separate initial_value)
        (2, 10, (False, True, False), True),
        (3, 10, (False,), False),
    ],
    ids=['3x3-operator-matrix', '1x1-operator-matrix'],
)
def matrixsystem(
    request,
) -> tuple[LinearOperatorMatrix, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
    """system Hx=b with linear and self-adjoint H as LinearOperatorMatrix."""
    rng = RandomGenerator(seed=456)
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
        initial_value = tuple(rng.rand_like(vector) for vector in vectors)
    else:
        initial_value = None
    return (operator_matrix, right_hand_side, tuple(vectors), initial_value)


def test_cg_solution(system) -> None:
    """Test if CG delivers accurate solution."""
    operator, right_hand_side, solution, initial_value = system
    (cg_solution,) = cg(operator, right_hand_side, initial_value=initial_value, max_iterations=256)
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
    (cg_solution,) = cg(
        operator, right_hand_side, initial_value=solution, max_iterations=10, tolerance=1e-4, callback=callback
    )
    assert (cg_solution == solution).all()


@pytest.mark.parametrize('max_iterations', [1, 2, 3, 5])
@pytest.mark.parametrize('use_preconditioner', [True, False], ids=['with preconditioner', 'without preconditioner'])
def test_compare_cg_to_scipy(system, max_iterations: int, use_preconditioner: bool) -> None:
    """Test if our implementation is close to the one of scipy."""
    operator, right_hand_side, _, initial_value = system

    if operator.matrix.ndim == 2:
        operator_sp = operator.matrix.numpy()
    else:
        operator_sp = scipy.linalg.block_diag(*operator.matrix.numpy())
    if use_preconditioner:
        ilu = scipy.sparse.linalg.spilu(scipy.sparse.csc_matrix(operator_sp), drop_tol=0.05)
        preconditioner_sp = scipy.sparse.linalg.LinearOperator(
            operator_sp.shape, lambda x: ilu.solve(x), dtype=operator_sp.dtype
        )
        preconditioner = lambda x: (torch.as_tensor(ilu.solve(x.flatten().numpy())).reshape(x.shape),)  #  # noqa: E731
    else:
        preconditioner_sp = preconditioner = None

    (scipy_solution, _) = scipy.sparse.linalg.cg(
        operator_sp,
        right_hand_side.flatten().numpy(),
        x0=None if initial_value is None else initial_value.flatten().numpy(),
        maxiter=max_iterations,
        atol=0,
        M=preconditioner_sp,
    )
    cg_solution_scipy = scipy_solution.reshape(right_hand_side.shape)
    (cg_solution,) = cg(
        operator,
        right_hand_side,
        initial_value=initial_value,
        max_iterations=max_iterations,
        tolerance=0,
        preconditioner_inverse=preconditioner,
    )

    tol = 1e-6 if cg_solution.dtype.to_real() == torch.float64 else 5e-3

    torch.testing.assert_close(cg_solution, torch.tensor(cg_solution_scipy), atol=tol, rtol=tol)


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
    operator, right_hand_side, _, _ = system

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
        (result,) = cg(operator, right_hand_side, initial_value=initial_value, tolerance=0, max_iterations=5)
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
