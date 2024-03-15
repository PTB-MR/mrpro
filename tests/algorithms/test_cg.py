"""Tests for the conjugate gradient method."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import pytest
import scipy
import torch
from mrpro.algorithms import cg
from mrpro.operators import EinsumOp
from scipy.sparse.linalg import cg as cg_scp
from tests import RandomGenerator


@pytest.fixture(
    params=[
        (1, 32, True),  # (batch-size, vector-size, flag for complex-valued system or not)
        (1, 32, False),
        (4, 32, True),
        (4, 32, False),
    ]
)
def system(request):
    """Generate data for creating a system Hx=b with linear and self-adjoint
    H."""
    random_generator = RandomGenerator(seed=0)

    # get parameters for the test
    batchsize, vectorsize, complex_valued = request.param

    # if batchsize=1, it corresponds to one linear system; for batchsize>1, multiple systems
    # are considered simultaneously
    matrix_shape: tuple[int, int, int] = (batchsize, vectorsize, vectorsize)
    vector_shape: tuple[int, int, int] = (batchsize, vectorsize)

    if complex_valued:
        matrix = random_generator.complex64_tensor(size=matrix_shape, high=1.0)
    else:
        matrix = random_generator.float32_tensor(size=matrix_shape, low=-1.0, high=1.0)

    # make sure H is self-adjoint
    self_adjoint_matrix = matrix.mH @ matrix

    # construct matrix multiplication as LinearOperator
    operator = EinsumOp(self_adjoint_matrix)

    # create ground-truth data and right-hand side of the system
    if complex_valued:
        vector = random_generator.complex64_tensor(size=vector_shape, high=1.0)
    else:
        vector = random_generator.float32_tensor(size=vector_shape, low=-1.0, high=1.0)

    (right_hand_side,) = operator(vector)

    return operator, right_hand_side, vector


def test_cg_convergence(system):
    """Test if CG delivers accurate solution."""

    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, solution = system

    initial_value = torch.ones_like(solution)
    cg_solution = cg(operator, right_hand_side, initial_value=initial_value, max_iterations=256)

    # test if solution is accurate
    torch.testing.assert_close(cg_solution, solution, rtol=5e-3, atol=5e-3)


def test_cg_stopping_after_one_iteration(system):
    """Test if cg stops after one iteration if the ground-truth is the initial
    guess."""
    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, solution = system

    # callback function; should not be called since cg should exit for loop
    def callback(solution):
        pytest.fail('CG did immediately not exit')

    # the test should fail if we reach the callback
    xcg_one_iteration = cg(
        operator, right_hand_side, initial_value=solution, max_iterations=10, tolerance=1e-4, callback=callback
    )
    assert (xcg_one_iteration == solution).all()


def test_compare_cg_to_scipy(system):
    """Test if our implementation is close to the one of scipy."""
    # create operator, right-hand side and ground-truth data
    operator, right_hand_side, _ = system

    # generate invalid initial value
    initial_value = torch.zeros_like(right_hand_side)

    # if batchsize>1, construct H = diag(H1,...,H_batchsize)
    # and b=[b1,...,b_batchsize]^T, otherwise just take the matrix
    matrix_np = scipy.linalg.block_diag(*operator.matrix.numpy())

    # choose zero tolerance to avoid exiting the for loop in the cg
    tolerance = 0.0

    # test for different maximal number of iterations
    for max_iterations in [1, 2, 4, 8]:
        # run our cg and scipy's cg and compare the results
        (xcg_scp, _) = cg_scp(
            matrix_np,
            right_hand_side.flatten().numpy(),
            x0=initial_value.flatten().numpy(),
            maxiter=max_iterations,
            atol=tolerance,
        )
        cg_solution_scipy = xcg_scp.reshape(right_hand_side.shape)
        cg_solution_torch = cg(
            operator,
            right_hand_side,
            initial_value=initial_value,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        torch.testing.assert_close(cg_solution_torch, torch.tensor(cg_solution_scipy), atol=1e-5, rtol=1e-5)


def test_invalid_shapes(system):
    """Test if CG throws error in case of shape-mismatch."""
    # create operator, right-hand side and ground-truth data
    h_operator, right_hand_side, _ = system

    # generate invalid initial point due to shape mismatch of
    # right_hand_side and input
    initial_value = torch.zeros(
        h_operator.matrix.shape[-1] + 1,
    )
    with pytest.raises(ValueError, match='match'):
        cg(h_operator, right_hand_side, initial_value=initial_value, max_iterations=10)


def test_callback(system):
    """Test if the callback function is called if a callback function is set."""
    # create operator, right-hand side
    h_operator, right_hand_side, _ = system

    # callback function; if the function is called during the iterations, the
    # test is successful
    def callback(solution):
        assert True

    cg(h_operator, right_hand_side, callback=callback)
