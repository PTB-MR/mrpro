"""Operator enforcing constraints by variable transformations."""

from collections.abc import Sequence

import torch

from mr2.operators.EndomorphOperator import EndomorphOperator, endomorph


def sigmoid(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Beta scaled sigmoid function."""
    return torch.nn.functional.sigmoid(beta * x)


def sigmoid_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of `sigmoid`."""
    return torch.logit(x) / beta


def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Beta scaled softplus function."""
    return -(1 / beta) * torch.nn.functional.logsigmoid(-beta * x)


def softplus_inverse(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Inverse of `softplus`."""
    return x + torch.log(-torch.expm1(-beta * x)) / beta


class ConstraintsOp(EndomorphOperator):
    """Transformation to map real-valued tensors to certain ranges."""

    def __init__(
        self,
        bounds: Sequence[tuple[float | None, float | None]],
        beta_sigmoid: float = 1.0,
        beta_softplus: float = 1.0,
    ) -> None:
        """Initialize a constraint operator.

        The operator maps real-valued tensors to certain ranges. The transformation is applied element-wise.
        The transformation is defined by the bounds. The bounds are applied in the order of the input tensors.
        If there are more input tensors than bounds, the remaining tensors are passed through without transformation.

        If an input tensor is bounded from below AND above, a sigmoid transformation is applied.
        If an input tensor is bounded from below OR above, a softplus transformation is applied.

        If an input is complex valued, the bounds are to the real and imaginary parts separately,
        i.e., for bounds :math:`(a, b)`, the complex number is constrained to a rectangle in the complex plane
        with corners :math:`(a+ai, a+bi, b+ai, b+bi)`.

        Parameters
        ----------
        bounds
            Sequence of `(lower_bound, upper_bound)` values. If a bound is `None`, the value is not constrained.
            If a lower bound is `-inf`, the value is not constrained from below. If an upper bound is `+inf`,
            the value is not constrained from above.
            If the bounds are set to `(None, None)` or `(-inf, +inf)`, the value is not constrained at all.
        beta_sigmoid
            beta parameter for the sigmoid transformation (used if an input has two bounds).
            A higher value leads to a steeper sigmoid.
        beta_softplus
            parameter for the softplus transformation (used if an input is either bounded from below or above).
            A higher value leads to a steeper softplus.

        Raises
        ------
        ValueError
            If the lower bound is greater than the upper bound.
        ValueError
            If the a bound is nan.
        ValueError
            If the parameter beta_sigmoid and beta_softplus are not greater than zero.
        """
        super().__init__()

        if beta_sigmoid <= 0:
            raise ValueError(f'parameter beta_sigmoid must be greater than zero; given {beta_sigmoid}')
        if beta_softplus <= 0:
            raise ValueError(f'parameter beta_softplus must be greater than zero; given {beta_softplus}')

        self.beta_sigmoid = beta_sigmoid
        self.beta_softplus = beta_softplus

        self.lower_bounds = tuple(torch.as_tensor(-torch.inf if lb is None else lb) for (lb, ub) in bounds)
        self.upper_bounds = tuple(torch.as_tensor(torch.inf if ub is None else ub) for (lb, ub) in bounds)

        for lb, ub in zip(self.lower_bounds, self.upper_bounds, strict=True):
            if lb.isnan():
                raise ValueError('nan is invalid as lower bound.')
            if ub.isnan():
                raise ValueError('nan is invalid as upper bound.')
            if lb >= ub:
                raise ValueError(
                    'bounds should be ( (a1,b1), (a2,b2), ...) with ai < bi if neither ai or bi is None;'
                    f'\nbound tuple {lb, ub} is invalid as the lower bound is higher than the upper bound',
                )

    def _apply_forward(self, item: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        """Apply the forward transformation to the input tensor."""
        if item.dtype.is_complex:
            real = self._apply_forward(item.real, lb, ub)
            imag = self._apply_forward(item.imag, lb, ub)
            return torch.complex(real, imag)

        if not lb.isneginf() and not ub.isposinf():
            # bounds are (lb,ub)
            return lb + (ub - lb) * sigmoid(item, beta=self.beta_sigmoid)

        if not lb.isneginf():
            # bounds are (lb,inf)
            return lb + softplus(item, beta=self.beta_softplus)

        if not ub.isposinf():
            # bounds are (-inf,ub)
            return ub - softplus(-item, beta=self.beta_softplus)

        return item  # unconstrained case

    def _apply_inverse(self, item: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
        if item.dtype.is_complex:
            real = self._apply_inverse(item.real, lb, ub)
            imag = self._apply_inverse(item.imag, lb, ub)
            return torch.complex(real, imag)

        if not lb.isneginf() and not ub.isposinf():
            # bounds are (lb,ub)
            return sigmoid_inverse((item - lb) / (ub - lb), beta=self.beta_sigmoid)

        if not lb.isneginf():
            # bounds are (lb,inf)
            return softplus_inverse(item - lb, beta=self.beta_softplus)

        if not ub.isposinf():
            # bounds are (-inf,ub)
            return -softplus_inverse(-(item - ub), beta=self.beta_softplus)

        return item  # unconstrained case

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Transform tensors to chosen range.

        Applies element-wise transformations to map input tensors to specified bounds.
        - If bounded below and above: uses a sigmoid transformation.
        - If bounded below or above: uses a softplus transformation.
        - If complex: applies transformation to real and imaginary parts separately.
        - If more input tensors than bounds: remaining tensors pass through unchanged.

        Parameters
        ----------
        *x
            One or more input tensors to be transformed.

        Returns
        -------
            Transformed tensors, with values mapped to the ranges defined by the bounds.
        """
        return super().__call__(*x)

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of ConstraintsOp.

        .. note::
            Prefer calling the instance of the ConstraintsOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        x_constrained = [
            self._apply_forward(item, lb, ub)
            for item, lb, ub in zip(x, self.lower_bounds, self.upper_bounds, strict=False)
        ]
        # if there are more inputs than bounds, pass on the remaining inputs without transformation
        x_constrained.extend(x[len(x_constrained) :])
        return tuple(x_constrained)

    @endomorph
    def invert(self, *x_constrained: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Reverses the variable transformation.

        Parameters
        ----------
        x_constrained
            transformed tensors with values in the range defined by the bounds

        Returns
        -------
            tensors in the domain with no bounds
        """
        x = [
            self._apply_inverse(item, lb, ub)
            for item, lb, ub in zip(x_constrained, self.lower_bounds, self.upper_bounds, strict=False)
        ]
        # if there are more inputs than bounds, pass on the remaining inputs without transformation
        x.extend(x_constrained[len(x) :])

        return tuple(x)

    @property
    def inverse(self) -> 'InverseConstraintOp':
        """Return the inverse of the constraint operator."""
        return InverseConstraintOp(self)


class InverseConstraintOp(EndomorphOperator):
    """Inverse of a constraint operator."""

    def __init__(self, constraints_op: ConstraintsOp) -> None:
        """Initialize the inverse constraint operator.

        Parameters
        ----------
        constraints_op
            The constraint operator to invert.
        """
        super().__init__()
        self.constraints_op = constraints_op

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the inverse of the constraint operator.

        This reverses the transformation applied by the corresponding `ConstraintsOp`,
        mapping values from their constrained ranges back to the unbounded domain.

        Parameters
        ----------
        *x
            One or more input tensors, assumed to be in the constrained ranges.

        Returns
        -------
            Tensors with the inverse transformation applied, mapped back to the unbounded domain.
        """
        return super().__call__(*x)

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply forward of InverseConstraintOp.

        .. note::
            Prefer calling the instance of the InverseConstraintOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self.constraints_op.invert(*x)

    @endomorph
    def invert(self, *x_constrained: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply constraint operator."""
        return self.constraints_op.forward(*x_constrained)

    @property
    def inverse(self) -> 'ConstraintsOp':
        """Return the constraint operator."""
        return self.constraints_op
