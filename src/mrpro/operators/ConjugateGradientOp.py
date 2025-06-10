"""Conjugate gradient operator."""

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import torch
from torch.autograd.function import once_differentiable

from mrpro.algorithms.optimizers.cg import cg
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.LinearOperatorMatrix import LinearOperatorMatrix

LinearOperatorFactory = Callable[..., LinearOperator]
LinearOperatorMatrixFactory = Callable[..., LinearOperatorMatrix]
T = TypeVar('T', torch.Tensor, bool)


class ConjugateGradientCTX(torch.autograd.function.FunctionCtx):
    """Only used for type hinting."""

    saved_tensors: tuple[torch.Tensor, ...]
    needs_input_grad: tuple[bool, ...]
    len_solution: int
    tolerance: float
    max_iterations: int
    rhs_factory: Callable[..., tuple[torch.Tensor, ...]]
    operator_factory: Callable[..., LinearOperatorMatrix | LinearOperator]


class ConjugateGradientFunction(torch.autograd.Function):
    """Autograd function for the regularized least squares operator."""

    if TYPE_CHECKING:

        @classmethod
        def apply(
            cls,
            operator_factory: Callable[..., LinearOperatorMatrix | LinearOperator],
            rhs_factory: Callable[..., tuple[torch.Tensor, ...]],
            *inputs: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            """Apply the function. Required for mypy."""
            return super().apply(operator_factory, rhs_factory, *inputs)

    @staticmethod
    def forward(
        ctx: ConjugateGradientCTX,
        operator_factory: Callable[..., LinearOperatorMatrix | LinearOperator],
        rhs_factory: Callable[..., tuple[torch.Tensor, ...]],
        *inputs: torch.Tensor,
        initial_value: tuple[torch.Tensor, ...] | None = None,
        max_iterations: int = 10000,
        tolerance: float = 1e-7,
    ) -> tuple[torch.Tensor, ...]:
        """Forward pass of the conjugate gradient operator."""
        operator = operator_factory(*inputs)
        rhs = rhs_factory(*inputs)
        rhs_norm = sum((r.abs().square().sum() for r in rhs), torch.tensor(0.0)).sqrt().item()
        fwd_tol = tolerance * max(rhs_norm, 1e-6)  # clip in case rhs is 0
        if isinstance(operator, LinearOperator):
            if len(rhs) != 1:
                raise ValueError('LinearOperator requires a single right-hand side tensor.')
            if initial_value is not None and len(initial_value) != 1:
                raise ValueError('LinearOperator requires a single initial value tensor.')
            solution: tuple[torch.Tensor, ...] = cg(
                operator, rhs, initial_value=initial_value, tolerance=fwd_tol, max_iterations=max_iterations
            )
        else:
            solution = cg(operator, rhs, initial_value=initial_value, tolerance=fwd_tol, max_iterations=max_iterations)
        ctx.save_for_backward(*solution, *inputs)
        ctx.len_solution = len(solution)
        ctx.tolerance = tolerance
        ctx.max_iterations = max_iterations
        ctx.rhs_factory = rhs_factory
        ctx.operator_factory = operator_factory
        return solution

    @staticmethod
    @once_differentiable
    def backward(ctx: ConjugateGradientCTX, *grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Backward pass of the conjugate gradient operator."""
        solution, inputs = (
            ctx.saved_tensors[: ctx.len_solution],
            ctx.saved_tensors[ctx.len_solution :],
        )
        with torch.enable_grad():
            rhs = ctx.rhs_factory(*inputs)
            operator = ctx.operator_factory(*inputs)
        inputs_with_grad = tuple(x for x, need_grad in zip(inputs, ctx.needs_input_grad[2:], strict=True) if need_grad)
        if inputs_with_grad:
            rhs_norm = sum((r.abs().square().sum() for r in grad_output), torch.tensor(0.0)).sqrt().item()
            bwd_tol = ctx.tolerance * max(rhs_norm, 1e-6)  # clip in case rhs is 0
            with torch.no_grad():
                if isinstance(operator, LinearOperatorMatrix):
                    z = cg(operator.H, grad_output, tolerance=bwd_tol, max_iterations=ctx.max_iterations)
                else:
                    z = cg(operator.H, grad_output[0], tolerance=bwd_tol, max_iterations=ctx.max_iterations)
            if any(zi.isnan().any() for zi in z):
                raise RuntimeError('NaN in ConjugateGradientFunction.backward')
            with torch.enable_grad():
                residual = tuple(r - ax for r, ax in zip(rhs, operator(*(s.detach() for s in solution)), strict=True))
            grads = torch.autograd.grad(outputs=residual, inputs=inputs_with_grad, grad_outputs=z, allow_unused=True)
            grad_iter = iter(grads)
        else:
            grad_iter = iter(())

        grad_input = tuple(next(grad_iter) if need else None for need in ctx.needs_input_grad[2:])
        return (None, None, *grad_input)  # operator_factory, rhs_factory, *inputs


class ConjugateGradientOp(torch.nn.Module):
    r"""Solves a linear positive semidefinite system with the conjugate gradient method.

    Solves :math: `A x = b` where :math:`A` is a linear operator , :math:`b` is a  tensor or a tuple of tensors.

    The operator is autograd differentiable using implicit differentiation.
    If this is not needed, consider using `mrpro.algorithms.optimizers.cg` directly.
    """

    def __init__(
        self,
        operator_factory: Callable[..., LinearOperatorMatrix | LinearOperator],
        rhs_factory: Callable[..., tuple[torch.Tensor, ...]],
        implicit_backward: bool = True,
        tolerance: float = 1e-8,
        max_iterations: int = 10000,
    ):
        r"""Initialize a conjugate gradient operator.

        Both the operator and the right-hand side are given as factory functions.
        The arguments given to the operator when calling it are passed to the factory functions.

        **Example: Regularized Least Squares**

        Consider the regularized least squares problem:
        :math:`\min_x \|A x - b\|_2^2 + \alpha \|x - x_0\|_2^2`.

        The normal equations are :math:`(A^H A + \alpha I) x = A^H b + \alpha x_0`.
        This can be solved using the ConjugateGradientOp as follows:
        .. code-block:: python
            operator_factory = lambda alpha, x0, b: A.gram + alpha
            rhs_factory = lambda alpha, x0, b: A.H(b)[0] + alpha * x0
            op = ConjugateGradientOp(operator_factory, rhs_factory)
            solution = op(alpha, x0, b)

        Parameters
        ----------
        operator_factory
            A factory function that returns the operator :math:`A`.
            Should return either a `LinearOperatorMatrix` or a `LinearOperator`.
        rhs_factory
            A factory function that returns the right-hand side :math:`b`
            Should return a tuple of tensors.
        implicit_backward
            If `True`, the backward pass is done using implicit differentiation.
            If `False`, the backward pass is done using unrolling the CG loop.
        tolerance
            The tolerance for the conjugate gradient method. The tolerance is relative
            to the norm of the right-hand side. The same relative tolerance is used in the
            backward pass if using implicit differentiation.
        max_iterations
            The maximum number of iterations for the conjugate gradient method.
            The same maximum number of iterations is used in the backward pass if using
            implicit differentiation.

        .. warning::
            If implicit_backward is `True`, the problem has to converge, otherwise the backward
            will be wrong. `tolerance` and `max_iter` should be chosen accordingly.
        """
        super().__init__()
        self.operator_factory = operator_factory
        self.rhs_factory = rhs_factory
        self.implicit_backward = implicit_backward
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def forward(
        self,
        *parameters: torch.Tensor,
        initial_value: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        r"""Solve the linear system using the conjugate gradient method.

        Parameters
        ----------
        parameters
            The parameters passed to the operator and right-hand side factory functions.
        initial_value
            The initial value for the conjugate gradient method.
            If `None`, the initial value is set to zero.
        """
        if self.implicit_backward:
            solution = ConjugateGradientFunction.apply(self.operator_factory, self.rhs_factory, *parameters)
        else:  # unrolled CG
            op = self.operator_factory(*parameters)
            rhs = self.rhs_factory(*parameters)
            rhs_norm = sum((r.abs().square().sum() for r in rhs), torch.tensor(0.0)).sqrt().item()
            fwd_tol = self.tolerance * rhs_norm
            if isinstance(op, LinearOperator):
                if len(rhs) != 1:
                    raise ValueError('LinearOperator requires a single right-hand side tensor.')
                if initial_value is not None and len(initial_value) != 1:
                    raise ValueError('LinearOperator requires a single initial value tensor.')
                solution = cg(
                    op, rhs, initial_value=initial_value, tolerance=fwd_tol, max_iterations=self.max_iterations
                )
            else:
                solution = cg(
                    op, rhs, initial_value=initial_value, tolerance=fwd_tol, max_iterations=self.max_iterations
                )
        return solution
