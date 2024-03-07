"""Conjugate Gradient for linear systems with self-adjoint linear operator."""

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


from collections.abc import Callable

import torch

from mrpro.operators import LinearOperator


def conjugate_gradient(
    operator: LinearOperator,
    right_hand_side: torch.Tensor,
    starting_value: torch.Tensor | None = None,
    max_iterations: int = 128,
    tolerance: float = 1e-4,
    callback: Callable | None = None,
) -> torch.Tensor:
    """CG for solving a linear system Hx=b.

    Thereby, H is a linear self-adjoint operator, b is the right-hand-side
    of the system and x is the sought solution.

    Note that this implementation allows for simultaneously solving a batch of B problems
    of the form
        H_i x_i = b_i,      i=1,...,B.
    Thereby, the underlying assumption is that the considered problem is H x = b with
        H:= diag(H_1, ..., H_B), b:= [b_1, ..., b_B]^T.
    Thus, if all H_i are self-adjoint, so is H and the CG can be applied.
    Note however, that the accuracy of the obtained solutions might vary among the different
    problems.

    Also, note that if the condition of H is very large, a small residual does not necessarily
    imply that the solution is accurate.

    Parameters
    ----------
    operator
        self-adjoint operator (named H above)
    right_hand_side
        right-hand-side of the system (named b above)
    starting_value, optional
        starting value of the iteration; if None, it will be set to b
    max_iterations, optional
        maximal number of iterations
    tolerance, optional
        tolerance for the residual
    callback, optional
        user-provided function to be called at each iteration

    Returns
    -------
        an approximate solution of the linear system Hx=b
    """

    # throw an error if the shapes of the starting value and the right-hand side
    # do not coincide
    if starting_value is not None:
        if starting_value.shape != right_hand_side.shape:
            raise ValueError(f'\nshapes {starting_value.shape, right_hand_side.shape} are incompatible')

    # initial residual
    residual = right_hand_side - operator(starting_value)[0] if starting_value is not None else right_hand_side.clone()

    # initialize conjugate direction
    conjugate_direction = residual.clone()

    # assign starting value to the solution
    solution = starting_value.clone() if starting_value is not None else right_hand_side.clone()

    # for the case where the residual is exactly zero
    if torch.vdot(residual.flatten(), residual.flatten()) == 0:
        return solution

    # squared tolerance;
    # (we will check ||residual||^2 < tolerance^2 instead of ||residual|| < tol
    # #to avoid the computation of the root for the norm)
    squared_tolerance = tolerance**2

    # dummy value the old squared norm of the residual;
    # #only required for initialization
    square_norm_residual_previous = None

    for i in range(max_iterations):

        # calculate the square norm of the residual
        residual_flat_new = residual.flatten()
        square_norm_residual_new = torch.vdot(residual_flat_new, residual_flat_new).real

        # check if the solution is already accurate enough
        if tolerance != 0:
            if square_norm_residual_new < squared_tolerance:
                return solution

        if i > 0:
            beta = square_norm_residual_new / square_norm_residual_previous
            conjugate_direction = residual + beta * conjugate_direction

        # update estimates of the solution and the residual
        (operator_conjugate_direction,) = operator(conjugate_direction)
        alpha = square_norm_residual_new / (
            torch.vdot(conjugate_direction.flatten(), operator_conjugate_direction.flatten())
        )
        solution += alpha * conjugate_direction
        residual -= alpha * operator_conjugate_direction

        square_norm_residual_previous = square_norm_residual_new

        if callback is not None:
            callback(solution)

    return solution
