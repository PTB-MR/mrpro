"""Conjugate Gradient for linear systems with self-adjoint linear operator."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import torch

from mrpro.operators.LinearOperator import LinearOperator


def cg(
    operator: LinearOperator,
    right_hand_side: torch.Tensor,
    initial_value: torch.Tensor | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable | None = None,
) -> torch.Tensor:
    """CG for solving a linear system Hx=b.

    Thereby, H is a linear self-adjoint operator, b is the right-hand-side
    of the system and x is the sought solution.

    Note that this implementation allows for simultaneously solving a batch of N problems
    of the form
        H_i x_i = b_i,      i=1,...,N.
    Thereby, the underlying assumption is that the considered problem is H x = b with
        H:= diag(H_1, ..., H_N), b:= [b_1, ..., b_N]^T.
    Thus, if all H_i are self-adjoint, so is H and the CG can be applied.
    Note however, that the accuracy of the obtained solutions might vary among the different
    problems.
    Note also that we don't test if the input operator is self-adjoint or not.

    Further, note that if the condition of H is very large, a small residual does not necessarily
    imply that the solution is accurate.

    Parameters
    ----------
    operator
        self-adjoint operator (named H above)
    right_hand_side
        right-hand-side of the system (named b above)
    initial_value, optional
        initial value of the algorithm; if None, it will be set to right_hand_side
    max_iterations, optional
        maximal number of iterations
    tolerance, optional
        tolerance for the residual; if set to zero, the maximal number of iterations
        is the only stopping criterion used to stop the cg
    callback, optional
        user-provided function to be called at each iteration

    Returns
    -------
        an approximate solution of the linear system Hx=b
    """
    if initial_value is not None and (initial_value.shape != right_hand_side.shape):
        raise ValueError(
            'Shapes of starting_value and right_hand_side must match,'
            f'got {initial_value.shape, right_hand_side.shape}'
        )

    # initial residual
    residual = right_hand_side - operator(initial_value)[0] if initial_value is not None else right_hand_side.clone()

    # initialize conjugate vector
    conjugate_vector = residual.clone()

    # assign starting value to the solution
    solution = initial_value.clone() if initial_value is not None else right_hand_side.clone()

    # for the case where the residual is exactly zero
    if torch.vdot(residual.flatten(), residual.flatten()) == 0:
        return solution

    # squared tolerance;
    # (we will check ||residual||^2 < tolerance^2 instead of ||residual|| < tol
    # to avoid the computation of the root for the norm)
    tolerance_squared = tolerance**2

    # dummy value. new value will be set in loop before first usage
    residual_norm_squared_previous = None

    for iteration in range(max_iterations):
        # calculate the square norm of the residual
        residual_flat = residual.flatten()
        residual_norm_squared = torch.vdot(residual_flat, residual_flat).real

        # check if the solution is already accurate enough
        if tolerance != 0 and (residual_norm_squared < tolerance_squared):
            return solution

        if iteration > 0:
            beta = residual_norm_squared / residual_norm_squared_previous
            conjugate_vector = residual + beta * conjugate_vector

        # update estimates of the solution and the residual
        (operator_conjugate_vector,) = operator(conjugate_vector)
        alpha = residual_norm_squared / (torch.vdot(conjugate_vector.flatten(), operator_conjugate_vector.flatten()))
        solution += alpha * conjugate_vector
        residual -= alpha * operator_conjugate_vector

        residual_norm_squared_previous = residual_norm_squared

        if callback is not None:
            callback(solution)

    return solution
