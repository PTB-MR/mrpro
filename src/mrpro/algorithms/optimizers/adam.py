"""ADAM for solving non-linear minimization problems."""

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

from collections.abc import Sequence

import torch
from torch.optim import Adam

from mrpro.operators.Operator import Operator


def adam(
    f: Operator[*tuple[torch.Tensor, ...], tuple[torch.Tensor]],
    initial_parameters: Sequence[torch.Tensor],
    max_iter: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0,
    amsgrad: bool = False,
    foreach: bool | None = None,
    maximize: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
) -> tuple[torch.Tensor, ...]:
    """Adam for non-linear minimization problems.

    Parameters
    ----------
    f
        scalar-valued function to be optimized
    initial_parameters
        Sequence (for example list) of parameters to be optimized.
        Note that these parameters will not be changed. Instead, we create a copy and
        leave the initial values untouched.
    max_iter
        maximum number of iterations
    lr
        learning rate, by default 1e-3
    betas
        coefficients used for computing running averages of gradient and its square,
        by default (0.9, 0.999)
    eps
        term added to the denominator to improve numerical stability, by default 1e-8
    weight_decay
        weight decay (L2 penalty), by default 0
    amsgrad
        whether to use the AMSGrad variant of this algorithm from the paper
        `On the Convergence of Adam and Beyond`, by default False
    foreach
        whether `foreach` implementation of optimizer is used, by default None
    maximize
        maximize the objective with respect to the params, instead of minimizing, by default False
    differentiable
        whether autograd should occur through the optimizer step. This is currently not implemented.
    fused
        whether the fused implementation (CUDA only) is used. Currently, torch.float64, torch.float32,
        torch.float16, and torch.bfloat16 are supported., by default None

    Returns
    -------
        list of optimized parameters
    """
    if not differentiable:
        parameters = [p.detach().clone().requires_grad_(True) for p in initial_parameters]
    else:
        # TODO: If differentiable is set, it is reasonable to expect that the result backpropagates to
        # initial parameters. This is currently not implemented (due to detach).
        raise NotImplementedError('Differentiable Optimization is not implemented')

    optim = Adam(
        params=parameters,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=foreach,
        maximize=maximize,
        differentiable=differentiable,
        fused=fused,
    )

    def closure():
        optim.zero_grad()
        (objective,) = f(*parameters)
        objective.backward()
        return objective

    # run adam
    for _ in range(max_iter):
        optim.step(closure)

    return tuple(parameters)
