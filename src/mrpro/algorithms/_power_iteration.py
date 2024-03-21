"""Power iteration for computing the largest eigenvalue and eigenvector of a linear function."""

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


def power_iteration(
    operator: Callable | LinearOperator, initial_value: torch.Tensor, max_iterations: int = 256, tolerance: float = 1e-6
):
    """Power iteration for computing the largest eigenvector and eigenvalue of a linear function.

    Parameters
    ----------
    operator
        operator whose largest eigenvalue and eigenvector are computed
    initial_value
        initial value to start the iteration; if chosen exactly as zero-tensor,
        we randomly generate an initial value
    max_iterations, optional
        maximal number of iterations, by default 256
    tolerance, optional
        tolerance used to determine when to stop the iteration, by default 1e-6;
        if set to zero, the maximal number of iterations is the only employed
        stopping criterion

    Returns
    -------
        the largest eigenvalue and eigenvector of the operator
    """
    # check if initial value is exactly zero; if yes, set it to a random value
    if (initial_value == 0.0).all():
        eigenvector_largest = torch.randn_like(initial_value)
    else:
        eigenvector_largest = initial_value / torch.linalg.norm(initial_value)

    for _ in range(max_iterations):
        eigenvector_largest_old = eigenvector_largest

        eigenvector_largest = operator(eigenvector_largest)[0]
        if isinstance(operator, LinearOperator):
            eigenvector_largest = eigenvector_largest[0]

        eigenvalue_largest = torch.vdot(eigenvector_largest_old.flatten(), eigenvector_largest.flatten())
        eigenvector_largest /= torch.linalg.norm(eigenvector_largest)

        if torch.linalg.norm(eigenvector_largest - eigenvector_largest_old) < tolerance:
            break

    return eigenvalue_largest, eigenvector_largest
