"""ADAM for solving non-linear minimization problems."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
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
from torch.optim import Adam


def adam(
    f,
    params: list,
    max_iter: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0,
    amsgrad: bool = False,
    foreach: bool | None = None,
    maximize: bool = False,
    capturable: bool = False,
    differentiable: bool = False,
    fused: bool | None = None,
) -> list[torch.Tensor]:
    """Adam for non-linear minimization problems.

    Parameters
    ----------
    f
        scalar-valued function to be optimized
    params
        list of parameters to be optimized
    lr, optional
        learning rate, by default 1e-3
    betas, optional
        coefficients used for computing running averages of gradient and its square,
        by default (0.9, 0.999)
    eps, optional
        term added to the denominator to improve numerical stability, by default 1e-8
    weight_decay, optional
        weight decay (L2 penalty), by default 0
    amsgrad, optional
        whether to use the AMSGrad variant of this algorithm from the paper
        `On the Convergence of Adam and Beyond`, by default False
    foreach, optional
        whether `foreach` implementation of optimizer is used, by default None
    maximize, optional
        maximize the objective with respect to the params, instead of minimizing, by default False
    capturable, optional
        whether this instance is safe to capture in a CUDA graph. Passing True can impair ungraphed
        performance, so if you don’t intend to graph capture this instance, leave it False, by default False
    differentiable, optional
        whether autograd should occur through the optimizer step in training. Otherwise, the step() function
        runs in a torch.no_grad() context. Setting to True can impair performance, so leave it False if you
        don’t intend to run autograd through this instance, by default False
    fused, optional
        whether the fused implementation (CUDA only) is used. Currently, torch.float64, torch.float32,
        torch.float16, and torch.bfloat16 are supported., by default None

    Returns
    -------
        list of optimized parameters
    """

    # define Adam routine
    adam_ = Adam(
        params=params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        foreach=foreach,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
    )

    def closure():
        adam_.zero_grad()
        (objective,) = f(*params)
        objective.backward()
        return objective

    # run adam
    for _ in range(max_iter):
        adam_.step(closure)

    return params
