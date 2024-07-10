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
from torch.optim import Adam, AdamW

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
    decoupled_weight_decay: bool = False,
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
        learning rate
    betas
        coefficients used for computing running averages of gradient and its square
    eps
        term added to the denominator to improve numerical stability
    weight_decay
        weight decay (L2 penalty)
    amsgrad
        whether to use the AMSGrad variant of this algorithm from the paper
        `On the Convergence of Adam and Beyond`
    decoupled_weight_decay
        whether to use Adam or AdamW [1]_

    Returns
    -------
        list of optimized parameters

    References
    ----------
    .. [1] Loshchilov I, Hutter F (2019) Decoupled Weight Decay Regularization. ICLR
            https://doi.org/10.48550/arXiv.1711.05101
    """
    parameters = [p.detach().clone().requires_grad_(True) for p in initial_parameters]

    optim: AdamW | Adam

    if not decoupled_weight_decay:
        optim = Adam(params=parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        optim = AdamW(params=parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    def closure():
        optim.zero_grad()
        (objective,) = f(*parameters)
        objective.backward()
        return objective

    # run adam
    for _ in range(max_iter):
        optim.step(closure)

    return tuple(parameters)
