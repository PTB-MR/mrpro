"""Linear Operators."""

from __future__ import annotations

import functools
import operator
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import cast, no_type_check

import torch
from typing_extensions import Any, Unpack, overload

import mrpro.operators
from mrpro.operators.Operator import Operator, OperatorComposition, OperatorSum, Tin2


class _AutogradWrapper(torch.autograd.Function):
    """Wrap forward and adjoint functions for autograd."""

    # If the forward and adjoint implementation are vmap-compatible,
    # the function can be marked as such to enable vmap support.
    generate_vmap_rule = True

    @staticmethod
    def forward(
        fw: Callable[[torch.Tensor], torch.Tensor],
        bw: Callable[[torch.Tensor], torch.Tensor],  # noqa: ARG004
        x: torch.Tensor,
    ) -> torch.Tensor:
        return fw(x)

    @staticmethod
    def setup_context(
        ctx: Any,  # noqa: ANN401
        inputs: tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor], torch.Tensor],
        output: torch.Tensor,
    ) -> torch.Tensor:
        ctx.fw, ctx.bw, x = inputs
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> tuple[None, None, torch.Tensor]:  # noqa: ANN401
        return None, None, _AutogradWrapper.apply(ctx.bw, ctx.fw, grad_output[0])

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> torch.Tensor:  # noqa: ANN401
        return _AutogradWrapper.apply(ctx.fw, ctx.bw, grad_inputs[-1])


class LinearOperator(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """General Linear Operator.

    LinearOperators have exactly one input tensors and one output tensor,
    and fulfill :math:`f(a*x + b*y) = a*f(x) + b*f(y)`
    with :math:`a`, :math:`b` scalars and :math:`x`, :math:`y` tensors.

    LinearOperators can be composed, added, multiplied, applied to tensors.
    LinearOperators have an `~LinearOperator.H` property that returns the adjoint operator,
    and a `~LinearOperator.gram` property that returns the Gram operator.

    Subclasses must implement the forward and adjoint methods.
    When subclassing, the `adjoint_as_backward` class attribute can be set to `True`::

            class MyOperator(LinearOperator, adjoint_as_backward=True):
                ...

    This will make pytorch use the adjoint method as the backward method of the forward,
    and the forward method as the backward method of the adjoint, avoiding the need to
    have differentiable forward and adjoint methods.
    """

    @no_type_check
    def __init_subclass__(cls, adjoint_as_backward: bool = False, **kwargs: Any) -> None:  # noqa: ANN401
        """Wrap the forward and adjoint functions for autograd.

        This will  wrap the forward and adjoint functions for autograd,
        and use the adjoint function as the backward function of the forward and vice versa.

        Parameters
        ----------
        adjoint_as_backward
            if True, the adjoint function is used as the backward function,
            else automatic differentiation of the forward function is used.
        kwargs
            additional keyword arguments, passed to the super class
        """
        if adjoint_as_backward and not hasattr(cls, '_saved_forward'):
            cls._saved_forward, cls._saved_adjoint = cls.forward, cls.adjoint
            cls.forward = functools.update_wrapper(
                lambda self, x: (
                    _AutogradWrapper.apply(lambda x: self._saved_forward(x)[0], lambda x: self._saved_adjoint(x)[0], x),
                ),
                cls.forward,
            )
            cls.adjoint = functools.update_wrapper(
                lambda self, x: (
                    _AutogradWrapper.apply(lambda x: self._saved_adjoint(x)[0], lambda x: self._saved_forward(x)[0], x),
                ),
                cls.adjoint,
            )
        super().__init_subclass__(**kwargs)

    @abstractmethod
    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the operator."""
        ...

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint operator.

        Obtains the adjoint of an instance of this operator as an `AdjointLinearOperator`,
        which itself is a an `LinearOperator` that can be applied to tensors.

        Note: ``linear_operator.H.H == linear_operator``
        """
        return AdjointLinearOperator(self)

    def operator_norm(
        self,
        initial_value: torch.Tensor,
        dim: Sequence[int] | None,
        max_iterations: int = 20,
        relative_tolerance: float = 1e-4,
        absolute_tolerance: float = 1e-5,
        callback: Callable[[torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Power iteration for computing the operator norm of the operator.

        Parameters
        ----------
        initial_value
            initial value to start the iteration; must be element of the domain.
            if the initial value contains a zero-vector for one of the considered problems,
            the function throws an `ValueError`.
        dim
            The dimensions of the tensors on which the operator operates. The choice of `dim` determines how
            the operator norm is inperpreted. For example, for a matrix-vector multiplication with a batched matrix
            tensor of shape `(batch1, batch2, row, column)` and a batched input tensor of shape `(batch1, batch2, row)`:

            * If `dim=None`, the operator is considered as a block diagonal matrix with batch1*batch2 blocks
              and the result is a tensor containing a single norm value (shape `(1, 1, 1)`).

            * If `dim=(-1)`, `batch1*batch2` matrices are considered, and for each a separate operator norm is computed.

            * If `dim=(-2,-1)`, `batch1` matrices with `batch2` blocks are considered, and for each matrix a
              separate operator norm is computed.

            Thus, the choice of `dim` determines implicitly determines the domain of the operator.
        max_iterations
            maximum number of iterations
        relative_tolerance
            absolute tolerance for the change of the operator-norm at each iteration;
            if set to zero, the maximal number of iterations is the only stopping criterion used to stop
            the power iteration.
        absolute_tolerance
            absolute tolerance for the change of the operator-norm at each iteration;
            if set to zero, the maximal number of iterations is the only stopping criterion used to stop
            the power iteration.
        callback
            user-provided function to be called at each iteration

        Returns
        -------
            An estimaton of the operator norm. Shape corresponds to the shape of the input tensor `initial_value`
            with the dimensions specified in `dim` reduced to a single value.
            The pointwise multiplication of `initial_value` with the result of the operator norm will always
            be well-defined.
        """
        if max_iterations < 1:
            raise ValueError('The number of iterations should be larger than zero.')

        dim = tuple(dim) if dim is not None else dim  # must be tuple or None for torch.sum

        norm_initial_value = torch.linalg.vector_norm(initial_value, dim=dim, keepdim=True)
        if not (norm_initial_value > 0).all():
            if dim is None:
                raise ValueError('The initial value for the iteration should be different from the zero-vector.')
            else:
                raise ValueError(
                    'Found at least one zero-vector as starting point. \
                    For each of the considered operators, the initial value for the iteration \
                    should be different from the zero-vector.'
                )

        # creaty dummy operator norm value that cannot be correct because by definition, the
        # operator norm is a strictly positive number. This ensures that the first time the
        # change between the old and the new estimate of the operator norm is non-zero and
        # thus prevents the loop from exiting despite a non-correct estimate.
        op_norm_old = torch.zeros(*tuple([1 for _ in range(initial_value.ndim)]), device=initial_value.device)

        gram = self.gram  # self.H@self

        vector = initial_value
        for _ in range(max_iterations):
            # apply the operator to the vector
            (vector_new,) = gram(vector)

            # compute estimate of the operator norm
            product = vector.real * vector_new.real
            if vector.is_complex() and vector_new.is_complex():
                product += vector.imag * vector_new.imag
            op_norm = product.sum(dim, keepdim=True).sqrt()

            # check if stopping criterion is fulfillfed; if not continue the iteration
            if (absolute_tolerance > 0 or relative_tolerance > 0) and torch.isclose(
                op_norm, op_norm_old, atol=absolute_tolerance, rtol=relative_tolerance
            ).all():
                break

            # normalize vector
            vector = vector_new / torch.linalg.vector_norm(vector_new, dim=dim, keepdim=True)
            op_norm_old = op_norm

            if callback is not None:
                callback(op_norm)

        return op_norm

    @overload
    def __matmul__(self, other: LinearOperator) -> LinearOperator: ...

    @overload
    def __matmul__(
        self, other: Operator[Unpack[Tin2], tuple[torch.Tensor,]]
    ) -> Operator[Unpack[Tin2], tuple[torch.Tensor,]]: ...

    def __matmul__(
        self,
        other: Operator[Unpack[Tin2], tuple[torch.Tensor,]] | LinearOperator,
    ) -> Operator[Unpack[Tin2], tuple[torch.Tensor,]] | LinearOperator:
        """Operator composition.

        Returns ``lambda x: self(other(x))``
        """
        if isinstance(other, mrpro.operators.IdentityOp):
            # neutral element of composition
            return self
        elif isinstance(self, mrpro.operators.IdentityOp):
            return other
        elif isinstance(self, mrpro.operators.ZeroOp) or isinstance(other, mrpro.operators.ZeroOp):
            # zero operator composition with any operator is zero
            return mrpro.operators.ZeroOp()
        elif isinstance(other, LinearOperator):
            # LinearOperator@LinearOperator is linear
            return LinearOperatorComposition(self, other)
        elif isinstance(other, Operator):
            # cast due to https://github.com/python/mypy/issues/16335
            return OperatorComposition(self, cast(Operator[Unpack[Tin2], tuple[torch.Tensor,]], other))
        return NotImplemented  # type: ignore[unreachable]

    def __radd__(self, other: torch.Tensor) -> LinearOperator:
        """Operator addition.

        Returns ``lambda x: self(x) + other*x``
        """
        return self + other

    @overload  # type: ignore[override]
    def __add__(self, other: LinearOperator | torch.Tensor) -> LinearOperator: ...

    @overload
    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor,]]
    ) -> Operator[torch.Tensor, tuple[torch.Tensor,]]: ...

    def __add__(
        self, other: Operator[torch.Tensor, tuple[torch.Tensor,]] | LinearOperator | torch.Tensor
    ) -> Operator[torch.Tensor, tuple[torch.Tensor,]] | LinearOperator:
        """Operator addition.

        Returns ``lambda x: self(x) + other(x)`` if other is a operator,
        ``lambda x: self(x) + other`` if other is a tensor
        """
        if isinstance(other, torch.Tensor):
            # tensor addition
            return LinearOperatorSum(self, mrpro.operators.IdentityOp() * other)
        elif isinstance(self, mrpro.operators.ZeroOp):
            # neutral element of addition
            return other
        elif isinstance(other, mrpro.operators.ZeroOp):
            # neutral element of addition
            return self
        elif isinstance(other, LinearOperator):
            # sum of LinearOperators is linear
            return LinearOperatorSum(self, other)
        elif isinstance(other, Operator):
            # for general operators
            return OperatorSum(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __mul__(self, other: torch.Tensor | complex) -> LinearOperator:
        """Operator elementwise left multiplication with tensor/scalar.

        Returns ``lambda x: self(x*other)``
        """
        if isinstance(other, complex | float | int):
            if other == 0:
                return mrpro.operators.ZeroOp()
            if other == 1:
                return self
            else:
                return LinearOperatorElementwiseProductLeft(self, other)
        elif isinstance(other, torch.Tensor):
            return LinearOperatorElementwiseProductLeft(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __rmul__(self, other: torch.Tensor | complex) -> LinearOperator:
        """Operator elementwise right multiplication with tensor/scalar.

        Returns ``lambda x: other*self(x)``
        """
        if isinstance(other, complex | float | int):
            if other == 0:
                return mrpro.operators.ZeroOp()
            if other == 1:
                return self
            else:
                return LinearOperatorElementwiseProductRight(self, other)
        elif isinstance(other, torch.Tensor):
            return LinearOperatorElementwiseProductRight(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __and__(self, other: LinearOperator) -> mrpro.operators.LinearOperatorMatrix:
        """Vertical stacking of two LinearOperators.

        ``A&B`` is a `~mrpro.operators.LinearOperatorMatrix` with two rows,
        with ``(A&B)(x) == (A(x), B(x))``.
        See `mrpro.operators.LinearOperatorMatrix` for more information.
        """
        if not isinstance(other, LinearOperator):
            return NotImplemented  # type: ignore[unreachable]
        operators = [[self], [other]]
        return mrpro.operators.LinearOperatorMatrix(operators)

    def __or__(self, other: LinearOperator) -> mrpro.operators.LinearOperatorMatrix:
        """Horizontal stacking of two LinearOperators.

        ``A|B`` is a `~mrpro.operators.LinearOperatorMatrix` with two columns,
        with ``(A|B)(x1,x2) == A(x1)+B(x2)``.
        See `mrpro.operators.LinearOperatorMatrix` for more information.
        """
        if not isinstance(other, LinearOperator):
            return NotImplemented  # type: ignore[unreachable]
        operators = [[self, other]]
        return mrpro.operators.LinearOperatorMatrix(operators)

    @property
    def gram(self) -> LinearOperator:
        """Gram operator.

        For a LinearOperator :math:`A`, the self-adjoint Gram operator is defined as :math:`A^H A`.

        .. note::
           This is the inherited default implementation.
        """
        return self.H @ self


class LinearOperatorComposition(LinearOperator):
    """Linear operator composition.

    Performs operator1(operator2(x))
    """

    def __init__(self, operator1: LinearOperator, operator2: LinearOperator) -> None:
        """Linear operator composition initialization.

        Returns ``lambda x: operator1(operator2(x))``

        Parameters
        ----------
        operator1
            outer operator
        operator2
            inner operator
        """
        super().__init__()
        self._operator1 = operator1
        self._operator2 = operator2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Linear operator composition."""
        return self._operator1(*self._operator2(x))

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the linear operator composition."""
        # (AB)^H = B^H A^H
        return self._operator2.adjoint(*self._operator1.adjoint(x))

    @property
    def gram(self) -> LinearOperator:
        """Gram operator."""
        # (AB)^H(AB) = B^H (A^H A) B
        return self._operator2.H @ self._operator1.gram @ self._operator2


class LinearOperatorSum(LinearOperator):
    """Linear operator addition."""

    _operators: list[LinearOperator]

    def __init__(self, operator1: LinearOperator, /, *other_operators: LinearOperator):
        """Linear operator addition initialization."""
        super().__init__()
        ops: list[LinearOperator] = []
        for op in (operator1, *other_operators):
            if isinstance(op, LinearOperatorSum):
                ops.extend(op._operators)
            else:
                ops.append(op)
        self._operators = cast(list[LinearOperator], torch.nn.ModuleList(ops))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Linear operator addition."""
        return (functools.reduce(operator.add, (op(x)[0] for op in self._operators)),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint of the linear operator addition."""
        # (A+B)^H = A^H + B^H
        return (functools.reduce(operator.add, (op.adjoint(x)[0] for op in self._operators)),)


class LinearOperatorElementwiseProductRight(LinearOperator):
    """Linear operator elementwise right multiplication with a tensor.

    Performs Tensor*LinearOperator(x)
    """

    def __init__(self, operator: LinearOperator, scalar: torch.Tensor | complex):
        """Linear operator elementwise right multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._scalar = scalar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Linear operator elementwise right multiplication."""
        (out,) = self._operator(x)
        return (out * self._scalar,)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint Operator elementwise multiplication with a tensor/scalar."""
        conj = self._scalar.conj() if isinstance(self._scalar, torch.Tensor) else self._scalar.conjugate()
        return self._operator.adjoint(x * conj)

    @property
    def gram(self) -> LinearOperator:
        """Gram Operator."""
        if isinstance(self._scalar, torch.Tensor):
            factor: torch.Tensor | complex = self._scalar.conj() * self._scalar
            if self._scalar.numel() > 1:
                # only scalars can be moved outside the linear operator
                return self._operator.H @ (factor * self._operator)
        else:
            factor = self._scalar.conjugate() * self._scalar
        return factor * self._operator.gram


class LinearOperatorElementwiseProductLeft(LinearOperator):
    """Operator elementwise left multiplication with a tensor.

    Performs LinearOperator(Tensor*x)
    """

    def __init__(self, operator: LinearOperator, scalar: torch.Tensor | complex) -> None:
        """Linear operator elementwise left multiplication initialization."""
        super().__init__()
        self._operator = operator
        self._scalar = scalar

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Linear operator elementwise left multiplication."""
        return self._operator(x * self._scalar)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Adjoint linear operator elementwise multiplication with a tensor/scalar."""
        conj = self._scalar.conj() if isinstance(self._scalar, torch.Tensor) else self._scalar.conjugate()
        return (self._operator.adjoint(x)[0] * conj,)

    @property
    def gram(self) -> LinearOperator:
        """Gram Operator."""
        conj = self._scalar.conj() if isinstance(self._scalar, torch.Tensor) else self._scalar.conjugate()
        return conj * self._operator.gram * self._scalar


class AdjointLinearOperator(LinearOperator):
    """Adjoint of a LinearOperator."""

    def __init__(self, operator: LinearOperator) -> None:
        """Initialize the adjoint of a LinearOperator."""
        super().__init__()
        self._operator = operator

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the original LinearOperator."""
        return self._operator.adjoint(x)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the adjoint, i.e. the original LinearOperator."""
        return self._operator.forward(x)

    @property
    def H(self) -> LinearOperator:  # noqa: N802
        """Adjoint of adjoint operator, i.e. original LinearOperator."""
        return self._operator
