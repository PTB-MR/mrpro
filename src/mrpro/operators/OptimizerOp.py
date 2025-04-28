"""Differentiable Minimization."""

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from typing_extensions import Any, TypeVar, TypeVarTuple, Unpack

from mrpro.algorithms.optimizers.cg import cg
from mrpro.algorithms.optimizers.lbfgs import lbfgs
from mrpro.operators.Operator import Operator

ArgumentType = TypeVarTuple('ArgumentType')
VariableType = TypeVar('VariableType', bound=tuple[torch.Tensor, ...])
ObjectiveType = Callable[..., tuple[torch.Tensor]] | Operator[Any, tuple[torch.Tensor]]
FactoryType = Callable[..., ObjectiveType]
OptimizeFunctionType = Callable[[Callable, VariableType], VariableType]

default_lbfgs = functools.partial(
    lbfgs,
    learning_rate=1.0,
    max_iterations=40,
    tolerance_change=1e-8,
    tolerance_grad=1e-7,
    history_size=20,
    line_search_fn='strong_wolfe',
)
"""LBFGS Optimizer"""


class OptimizeCtx(torch.autograd.function.FunctionCtx):
    """Rype hinting the CTX object."""

    factory: Callable[
        [Unpack[tuple[torch.Tensor, ...]]], Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor]]
    ]
    len_x: int
    needs_input_grad: tuple[bool, ...]
    saved_tensors: tuple[torch.Tensor, ...]


class OptimizeFunction(torch.autograd.Function):
    """Implicit Backward."""

    if TYPE_CHECKING:

        @classmethod
        def apply(
            cls,
            factory: Callable[..., Callable[..., tuple[torch.Tensor]]],
            initial_values: tuple[torch.Tensor, ...],
            optimize: Callable[
                [Callable[[*tuple[Unpack[tuple[torch.Tensor, ...]]]], tuple[torch.Tensor]], tuple[torch.Tensor, ...]],
                tuple[torch.Tensor, ...],
            ] = default_lbfgs,
            *parameters: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            """Apply the function. Only used for type hinting."""
            return super().apply(factory, initial_values, optimize, *parameters)

    @staticmethod
    def forward(
        ctx: OptimizeCtx,
        factory: Callable[
            [Unpack[tuple[torch.Tensor, ...]]], Callable[[Unpack[tuple[torch.Tensor, ...]]], tuple[torch.Tensor]]
        ],
        initial_values: tuple[torch.Tensor, ...],
        optimize: Callable[
            [Callable[[*tuple[Unpack[tuple[torch.Tensor, ...]]]], tuple[torch.Tensor]], tuple[torch.Tensor, ...]],
            tuple[torch.Tensor, ...],
        ] = default_lbfgs,
        *parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Optimize."""
        ctx.factory = factory

        parameters_ = tuple(p.detach().clone() for p in parameters if isinstance(p, torch.Tensor))
        initial_values_ = tuple(x.detach().requires_grad_(True) for x in initial_values if isinstance(x, torch.Tensor))
        f = factory(*parameters)
        xprime = optimize(f, initial_values)
        ctx.save_for_backward(*xprime, *parameters_)
        ctx.len_x = len(initial_values_)
        return xprime

    @staticmethod
    def backward(ctx: OptimizeCtx, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Calculate the backward pass using implicit differentiation."""
        xprime = tuple(xp.detach().clone().requires_grad_(True) for xp in ctx.saved_tensors[: ctx.len_x])
        parameters = ctx.saved_tensors[ctx.len_x :]
        parameters = tuple(
            p.detach().clone().requires_grad_(True) if ctx.needs_input_grad[i + 3] else p.detach()
            for i, p in enumerate(parameters)
        )
        dparams = [p for p in parameters if p.requires_grad]

        objective = ctx.factory(*parameters)

        def hvp(*v: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return torch.autograd.functional.vhp(lambda *x: objective(*x)[0], xprime, v=v)[1]

        hessian_inverse_grad = cg(hvp, grad_outputs, max_iterations=100, tolerance=1e-7)
        with torch.enable_grad():
            dobjective_dxprime = torch.autograd.grad(objective(*xprime), xprime, create_graph=True)
            # - d^2_obective / d_xprime d_params Hessian^-1_grad
            grad_params = list(torch.autograd.grad(dobjective_dxprime, dparams, hessian_inverse_grad))
        grad_inputs: list[torch.Tensor | None] = [None, None, None]  # factory, x0, optimize
        for need_grad in ctx.needs_input_grad[3:]:
            if need_grad:
                grad_inputs.append(-grad_params.pop(0))
            else:
                grad_inputs.append(None)

        return tuple(grad_inputs)


class OptimizerOp(Operator[Unpack[ArgumentType], VariableType]):
    """Differentiable Optimization Operator.

    One of the building blocks of PINQI [ZIMM2024]_
    Finds :math:`x^*=argmin_x f_p(x)

    References
    ----------
    .. [ZIMM2024] Zimmermann, Felix F., et al. (2024) PINQI. An End-to-End Physics-Informed Approach to Learned
       Quantitative MRI Reconstruction. IEEE TCI. https://doi.org/10.1109/TCI.2024.3388869
    """

    def __init__(
        self,
        factory: FactoryType,
        initializer: Callable[[Unpack[ArgumentType]], VariableType],
        optimize: OptimizeFunctionType = default_lbfgs,
    ):
        r"""Initialize a differentiable argmin solver.

        Parameters
        ----------
        factory
            Function, that given the parameters of the problem returns an objective function.
            The objective function should be a callable that takes the variable(s) as input and returns a scalar.
        initializer
            Function, that given the parameters of the problem returns a tuple of initial values for the variable(s)
        optimize
            Function used to perform the optimization, for example `lbfgs`.
            Use `functools.partial` to setup up all settings besides the objective function and the initial values.

        Example
        -------
            Solving :math:`\|q(x)-y\|^2 + \lambda*\|x-x_\mathrm{reg}\|^2` with
            :math:`y`, :math:`\lambda` and :math:`x_\mathrm{reg}` parameters. The solution :math:`x^*` should be
            differentiable with respect to these.

            Use::

                def factory(y, lambda, x_reg):
                    return L2squared(y)@q+lambda*L2squared(x_reg)
                def initializer(_y, _lambda, _xreg):
                    return (x_reg,)

        Returns
        -------
            The argmin `x^*`
        """
        super().__init__()
        self.factory = factory
        self.optimize = optimize
        self.initializer = initializer

    def forward(self, *parameters: Unpack[ArgumentType]) -> VariableType:
        """Find the argmin.

        Parameters
        ----------
        parameters
            Parameters of the argmin problem.
        """
        initial_values = self.initializer(*parameters)
        initial_values_ = tuple(x.clone() if any(x is p for p in parameters) else x for x in initial_values)
        result = OptimizeFunction.apply(
            self.factory, initial_values_, self.optimize, *cast(tuple[torch.Tensor, ...], parameters)
        )
        return cast(VariableType, result)
