"""LBFGS for solving non-linear minimization problems."""

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

from collections.abc import Sequence
from typing import Literal

import torch
from torch.optim import LBFGS

from mrpro.operators._Operator import Operator


def lbfgs(
    f: Operator[*tuple[torch.Tensor, ...], tuple[torch.Tensor]],
    initial_parameters: Sequence[torch.Tensor],
    lr: float = 1.0,
    max_iter: int = 100,
    max_eval: int | None = 100,
    tolerance_grad: float = 1e-07,
    tolerance_change: float = 1e-09,
    history_size: int = 10,
    line_search_fn: None | Literal['strong_wolfe'] = 'strong_wolfe',
) -> tuple[torch.Tensor, ...]:
    """LBFGS for non-linear minimization problems.

    Parameters
    ----------
    f
        scalar function to be minimized
    initial_parameters
        Sequence (for example list) of parameters to be optimized.
        Note that these parameters will not be changed. Instead, we create a copy and
        leave the initial values untouched.
    lr, optional
        learning rate
    max_iter, optional
        maximal number of iterations, by default 100
    max_eval, optional
        maximal number of evaluations of f per optimization step,
        by default 100
    tolerance_grad, optional
        termination tolerance on first order optimality,
        by default 1e-07
    tolerance_change, optional
        termination tolerance on function value/parameter changes, by default 1e-09
    history_size, optional
        update history size, by default 10
    line_search_fn, optional
        line search algorithm, either 'strong_wolfe' or None (meaning constant step size)
        by default "strong_wolfe"

    Returns
    -------
        list of optimized parameters
    """
    # TODO: remove after new pytorch release;
    if torch.tensor([torch.is_complex(p) for p in initial_parameters]).any():
        raise ValueError(
            "at least one tensor in 'params' is complex-valued; \
            \ncomplex-valued tensors will be allowed for lbfgs in future torch versions",
        )

    parameters = [p.detach().clone().requires_grad_(True) for p in initial_parameters]
    optim = LBFGS(
        params=parameters,
        lr=lr,
        history_size=history_size,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn,
    )

    def closure():
        optim.zero_grad()
        (objective,) = f(*parameters)
        objective.backward()
        return objective

    # run lbfgs
    optim.step(closure)

    return tuple(parameters)
