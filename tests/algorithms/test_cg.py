"""Tests for the conjugate gradient and biconjugate gradient methods."""

from collections.abc import Callable

import pytest
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import torch
from mr2.algorithms.optimizers import bicg, cg
from mr2.algorithms.optimizers.cg import CGStatus
from mr2.operators import EinsumOp, LinearOperator, LinearOperatorMatrix
from mr2.utils import RandomGenerator


@pytest.fixture(
    params=[  # (batch-size, vector-size, complex-valued system, separate initial_value)
        ((), 16, False, False),
        ((4,), 16, True, True),
    ],
    ids=[
        'real_single_noinit',
        'complex_batch',
    ],
)
def spd_system(
    request,
) -> tuple[LinearOperator, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None]:
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

    right_hand_side = operator(vector)
    if separate_initial_value:
        initial_value = (rng.rand_like(vector),)
    else:
        initial_value = None

    return operator, right_hand_side, (vector,), initial_value


@pytest.fixture(
    params=[  # (batch-size, vector-size, complex-valued system, separate initial_value)
        (2, 10, (False, True, False), True),
        (3, 10, (False,), False),
    ],
    ids=['3x3-operator-matrix', '1x1-operator-matrix'],
)
def spd_matrix_system(
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
            matrix = rng.complex128_tensor(size=matrix_shape, high=1.0)
            vector = rng.complex128_tensor(size=vector_shape, high=1.0)
        else:
            matrix = rng.float64_tensor(size=matrix_shape, low=-1.0, high=1.0)
            vector = rng.float64_tensor(size=vector_shape, low=-1.0, high=1.0)
        vectors.append(vector)
        operators.append(EinsumOp(matrix.mH @ matrix))  # make sure H is self-adjoint
    operator_matrix = LinearOperatorMatrix.from_diagonal(*operators)
    right_hand_side = operator_matrix(*vectors)
    if separate_initial_value:
        initial_value = tuple(rng.rand_like(vector) for vector in vectors)
    else:
        initial_value = None
    return (operator_matrix, right_hand_side, tuple(vectors), initial_value)


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_spd_solution(
    spd_system: tuple[LinearOperator, tuple[torch.Tensor], tuple[torch.Tensor], tuple[torch.Tensor] | None],
    algorithm: Callable,
) -> None:
    """Test if CG delivers accurate solution."""
    operator, right_hand_side, solution, initial_value = spd_system
    result = algorithm(operator, right_hand_side, initial_value=initial_value, max_iterations=256)
    torch.testing.assert_close(result, solution, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_spd_matrix_solution(
    spd_matrix_system: tuple[
        LinearOperatorMatrix, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None
    ],
    algorithm: Callable,
) -> None:
    """Test if CG delivers accurate solution for a LinearOperatorMatrix."""
    operator, right_hand_side, solution, initial_value = spd_matrix_system
    result = algorithm(
        operator,
        right_hand_side,
        initial_value=initial_value,
        max_iterations=1000,
        tolerance=1e-6,
    )
    torch.testing.assert_close(result, solution, rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_no_iteration(
    spd_system: tuple[LinearOperator, tuple[torch.Tensor], tuple[torch.Tensor], tuple[torch.Tensor] | None],
    algorithm: Callable,
) -> None:
    """Test if cg stops without performing any iterations if the ground-truth is the initial value."""
    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, solution, _ = spd_system

    # callback function; should not be called since we should stop before any iterations
    def callback(solution):
        pytest.fail('CG did not exit before performing any iterations')

    # the test should fail if we reach the callback
    result = algorithm(
        operator, right_hand_side, initial_value=solution, max_iterations=10, tolerance=1e-4, callback=callback
    )
    assert result == solution  # nothing should have happened


@pytest.mark.parametrize('algorithm', [cg, bicg])
@pytest.mark.parametrize('max_iterations', [1, 2, 3, 5])
@pytest.mark.parametrize('use_preconditioner', [True, False], ids=['with preconditioner', 'without preconditioner'])
def test_spd_compare_to_scipy(
    spd_system: tuple[EinsumOp, tuple[torch.Tensor], tuple[torch.Tensor], tuple[torch.Tensor] | None],
    max_iterations: int,
    use_preconditioner: bool,
    algorithm: Callable,
) -> None:
    """Test if our implementation is close to the one of scipy."""
    operator, right_hand_side, _, initial_value = spd_system

    if operator.matrix.ndim == 2:
        operator_sp = operator.matrix.numpy()
    else:
        operator_sp = scipy.linalg.block_diag(*operator.matrix.numpy())
    if use_preconditioner:
        ilu = scipy.sparse.linalg.spilu(scipy.sparse.csc_matrix(operator_sp), drop_tol=0.08)
        preconditioner_sp = scipy.sparse.linalg.LinearOperator(
            operator_sp.shape, lambda x: ilu.solve(x), dtype=operator_sp.dtype
        )
        preconditioner = lambda x: (torch.as_tensor(ilu.solve(x.flatten().numpy())).reshape(x.shape),)  #  # noqa: E731
    else:
        preconditioner_sp = preconditioner = None

    if algorithm == cg:
        sp_algorithm = scipy.sparse.linalg.cg
    else:
        sp_algorithm = scipy.sparse.linalg.bicgstab

    (scipy_result, _) = sp_algorithm(
        operator_sp,
        right_hand_side[0].flatten().numpy(),
        x0=None if initial_value is None else initial_value[0].flatten().numpy(),
        maxiter=max_iterations,
        atol=1e-6,
        M=preconditioner_sp,
    )
    scipy_result = scipy_result.reshape(right_hand_side[0].shape)

    (result,) = algorithm(
        operator,
        right_hand_side,
        initial_value=initial_value,
        max_iterations=max_iterations,
        tolerance=1e-6,
        preconditioner_inverse=preconditioner,
    )

    tol = 1e-6 if result.dtype.to_real() == torch.float64 else 5e-3
    torch.testing.assert_close(result, torch.tensor(scipy_result), atol=tol, rtol=tol)


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_invalid_shapes(
    spd_system: tuple[EinsumOp, torch.Tensor, torch.Tensor, torch.Tensor | None],
    algorithm: Callable,
) -> None:
    """Test if CG throws error in case of shape-mismatch."""
    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, *_ = spd_system

    # invalid initial value with mismatched shape
    bad_initial_value = torch.zeros(
        operator.matrix.shape[-1] + 1,
    )
    with pytest.raises(ValueError, match='match'):
        algorithm(operator, right_hand_side, initial_value=bad_initial_value, max_iterations=10)


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_callback(
    spd_system: tuple[EinsumOp, torch.Tensor, torch.Tensor, torch.Tensor | None],
    algorithm: Callable,
) -> None:
    """Test if the callback function is called if a callback function is set."""
    # create operator, right-hand side
    operator, right_hand_side, _, _ = spd_system

    # callback function; if the function is called during the iterations, the
    # test is successful
    def callback(cg_status: CGStatus) -> None:
        _, _, _ = cg_status['iteration_number'], cg_status['solution'][0], cg_status['residual'][0].norm()
        assert True

    algorithm(operator, right_hand_side, callback=callback)


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_callback_early_stop(
    spd_system: tuple[EinsumOp, torch.Tensor, torch.Tensor, torch.Tensor | None],
    algorithm: Callable,
) -> None:
    operator, right_hand_side, _, initial_value = spd_system
    """Check that when the callback function returns False the optimizer is stopped."""
    callback_check = 0

    # callback function that returns False to stop the algorithm
    def callback(solution):
        nonlocal callback_check
        callback_check += 1
        return False

    algorithm(operator, right_hand_side, initial_value=initial_value, max_iterations=100, callback=callback)
    assert callback_check == 1


@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_autograd(
    spd_system: tuple[EinsumOp, torch.Tensor, torch.Tensor, torch.Tensor | None],
    algorithm: Callable,
) -> None:
    """Test autograd through cg"""
    operator, right_hand_side, _, initial_value = spd_system
    right_hand_side[0].requires_grad_(True)
    with torch.autograd.detect_anomaly():
        (result,) = algorithm(operator, right_hand_side, initial_value=initial_value, max_iterations=5)
        result.abs().sum().backward()
    assert right_hand_side[0].grad is not None


@pytest.mark.cuda
@pytest.mark.parametrize('algorithm', [cg, bicg])
def test_cuda(
    spd_matrix_system: tuple[
        LinearOperatorMatrix, tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...] | None
    ],
    algorithm: Callable,
) -> None:
    """Test if CG works on CUDA."""
    operator, right_hand_side, solution, initial_value = spd_matrix_system
    right_hand_side = tuple(x.to('cuda') for x in right_hand_side)
    operator = operator.to('cuda')
    initial_value = tuple(x.to('cuda') for x in initial_value) if initial_value is not None else None
    solution = tuple(x.to('cuda') for x in solution)
    (result,) = algorithm(operator, right_hand_side, initial_value=initial_value, tolerance=1e-6, max_iterations=1000)
    assert all(x.is_cuda for x in result)
    torch.testing.assert_close(result, solution, rtol=5e-3, atol=5e-3)
