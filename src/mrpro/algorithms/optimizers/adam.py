"""ADAM for solving non-linear minimization problems."""

from collections.abc import Callable, Sequence

import torch
from torch.optim import Adam, AdamW

from mrpro.algorithms.optimizers.OptimizerStatus import OptimizerStatus
from mrpro.operators.Operator import OperatorType


def adam(
    f: OperatorType,
    initial_parameters: Sequence[torch.Tensor],
    max_iter: int,
    lr: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0,
    amsgrad: bool = False,
    decoupled_weight_decay: bool = False,
    callback: Callable[[OptimizerStatus], None] | None = None,
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
        weight decay (L2 penalty if decoupled_weight_decay is False)
    amsgrad
        whether to use the AMSGrad variant of this algorithm from the paper
        `On the Convergence of Adam and Beyond`
    decoupled_weight_decay
        whether to use Adam (default) or AdamW (if set to true) [LOS2019]_
    callback
        function to be called after each iteration


    Returns
    -------
        list of optimized parameters

    References
    ----------
    .. [LOS2019] Loshchilov I, Hutter F (2019) Decoupled Weight Decay Regularization. ICLR
       https://doi.org/10.48550/arXiv.1711.05101
    """
    parameters = tuple(p.detach().clone().requires_grad_(True) for p in initial_parameters)

    optim: AdamW | Adam

    if not decoupled_weight_decay:
        optim = Adam(params=parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
    else:
        optim = AdamW(params=parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    def closure():
        optim.zero_grad()
        (objective,) = f(*parameters)
        objective.backward()

        if callback is not None:
            callback({'solution': parameters, 'iteration_number': iteration})

        return objective

    # run adam
    for iteration in range(max_iter):  # noqa: B007
        optim.step(closure)

    return parameters
