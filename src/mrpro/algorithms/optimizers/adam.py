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
    r"""Adam for non-linear minimization problems.

    Adam [KING2015]_ (Adaptive Moment Estimation) is a first-order optimization algorithm that adapts learning rates
    for each parameter using estimates of the first and second moments of the gradients.

    The parameter update rule is:

    .. math::

        m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\
        v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \\
        \hat{m}_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
        \theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t

    where
    :math:`g_t` is the gradient at step :math:`t`,
    :math:`m_t` and :math:`v_t` are biased estimates of the first and second moments,
    :math:`\hat{m}_t` and :math:`\hat{v}_t` are bias-corrected estimates,
    :math:`\eta` is the learning rate,
    :math:`\epsilon` is a small constant for numerical stability,
    :math:`\beta_1` and :math:`\beta_2` are decay rates for the moment estimates.

    Steps of the Adam algorithm:

    1. Initialize parameters and moment estimates (:math:`m_0`, :math:`v_0`).
    2. Compute the gradient of the objective function.
    3. Compute bias-corrected estimates of the moments :math:`\hat{m}_t` and :math:`\hat{v}_t`.
    4. Update parameters using the adaptive step size.

    This function wraps PyTorch's :class:`torch.optim.Adam` and :class:`torch.optim.AdamW` implementations,
    supporting both standard Adam and decoupled weight decay regularization (AdamW) [LOS2019]_

    References
    ----------
    .. [KING2015] Kingma DP, Ba J (2015) Adam: A Method for Stochastic Optimization. ICLR.
       https://doi.org/10.48550/arXiv.1412.6980
    .. [LOS2019] Loshchilov I, Hutter F (2019) Decoupled Weight Decay Regularization. ICLR.
       https://doi.org/10.48550/arXiv.1711.05101
    .. [REDDI2019] Sashank J. Reddi, Satyen Kale, Sanjiv Kumar (2019) On the Convergence of Adam and Beyond. ICLR.
       https://doi.org/10.48550/arXiv.1904.09237

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
        weight decay (L2 penalty if `decoupled_weight_decay` is `False`)
    amsgrad
        whether to use the AMSGrad variant [REDDI2019]_
    decoupled_weight_decay
        whether to use Adam (default) or AdamW (if set to `True`) [LOS2019]_
    callback
        function to be called after each iteration

    Returns
    -------
        list of optimized parameters
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
