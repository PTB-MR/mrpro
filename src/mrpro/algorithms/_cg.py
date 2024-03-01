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

import torch

from mrpro.operators import LinearOperator


def cg(
    H: LinearOperator, b: torch.Tensor, x0: torch.Tensor | None = None, max_iter: int = 128, tol: float = 1e-4
) -> torch.Tensor:
    """CG for solving a linear system Hx=b, where H isself-adjoint.

    N.B. if the condition of H is very large, a small residual does not necessarily
    imply that the solution is accurate.

    Parameters
    ----------
    H
        Self-adjoint operator
    b
        right-hand-side of the system
    x0, optional
        starting value of the iteration; if not chosen, x0:=b
    max_iter, optional
        number of maximal iterations
    tol, optional
        tolerance for the residual

    Returns
    -------
        an approximate solution of the linear system Hx=b
    """

    if x0 is not None:
        if x0.shape != b.shape:
            raise ValueError(
                'CG solves linear systems Hx=b with self-adjoint operator;'
                'this implies that x0 has to have the same shape as b'
                f'\nshapes {x0.shape, b.shape} are incompatible'
            )

    # initial guess of the residual
    r = b - H(x0)[0] if x0 is not None else b.clone()

    # initialize p
    p = r.clone()

    sqnorm_b = torch.vdot(b.flatten(), b.flatten()).real
    if x0 is not None and torch.vdot(r.flatten(), r.flatten()).real <= tol**2 * sqnorm_b:
        return x0
    else:
        x = x0.clone() if x0 is not None else b.clone()

    for i in range(max_iter):
        rTr = torch.vdot(r.flatten(), r.flatten())
        (Hp,) = H(p)
        alpha = rTr / (torch.vdot(p.flatten(), Hp.flatten()))
        x = x + alpha * p
        r = r - alpha * Hp

        sqnorm_r = torch.vdot(r.flatten(), r.flatten()).real
        if sqnorm_r <= tol * sqnorm_b:
            return x

        if i < max_iter - 1:
            beta = sqnorm_r / rTr
            p = r + beta * p

    return x
