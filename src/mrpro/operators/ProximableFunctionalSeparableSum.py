"""Separable Sum of Proximable Functionals."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from functools import reduce
from typing import cast

import torch
from typing_extensions import Self, Unpack

from mrpro.operators.Functional import ProximableFunctional
from mrpro.operators.Operator import Operator


class ProximableFunctionalSeparableSum(Operator[Unpack[tuple[torch.Tensor, ...]], tuple[torch.Tensor]]):
    r"""Separabke Sum of Proximable Functionals.

    This is a separable sum of the functionals. The forward method returns the sum of the functionals
    evaluated at the inputs, :math:`\sum_i f_i(x_i)`.
    """

    functionals: tuple[ProximableFunctional, ...]

    def __init__(self, *functionals: ProximableFunctional) -> None:
        """Initialize the separable sum of proximable functionals.

        Parameters
        ----------
        functionals
            The proximable functionals to be summed.
        """
        super().__init__()
        self.functionals = functionals

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the functionals to the inputs.

        Parameters
        ----------
        x
            The inputs to the functionals

        Returns
        -------
            The sum of the functionals applied to the inputs
        """
        if len(x) != len(self.functionals):
            raise ValueError('The number of inputs must match the number of functionals.')
        result = reduce(operator.add, (f(xi)[0] for f, xi in zip(self.functionals, x, strict=True)))
        return (result,)

    def prox(self, *x: torch.Tensor, sigma: float | torch.Tensor = 1) -> tuple[torch.Tensor, ...]:
        """Apply the proximal operators of the functionals to the inputs.

        Parameters
        ----------
        x
            The inputs to the proximal operators
        sigma
            The scaling factor for the proximal operators

        Returns
        -------
            A tuple of the proximal operators applied to the inputs
        """
        prox_x = tuple(
            f.prox(xi, sigma)[0] for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)
        )
        return prox_x

    def prox_convex_conj(self, *x: torch.Tensor, sigma: float | torch.Tensor = 1) -> tuple[torch.Tensor, ...]:
        """Apply the proximal operators of the convex conjugate of the functionals to the inputs.

        Parameters
        ----------
        x
            The inputs to the proximal operators
        sigma
            The scaling factor for the proximal operators

        Returns
        -------
            A tuple of the proximal convex conjugate operators applied to the inputs
        """
        prox_convex_conj_x = tuple(
            f.prox_convex_conj(xi, sigma)[0]
            for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)
        )
        return prox_convex_conj_x

    def __or__(
        self,
        other: ProximableFunctional | ProximableFunctionalSeparableSum,
    ) -> Self:
        """Separable sum functionals."""
        if isinstance(other, ProximableFunctionalSeparableSum):
            return self.__class__(*self.functionals, *other.functionals)
        elif isinstance(other, ProximableFunctional):
            return self.__class__(*self.functionals, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __ror__(self, other: ProximableFunctional) -> Self:
        """Separable sum functionals."""
        if isinstance(other, ProximableFunctional):
            return self.__class__(other, *self.functionals)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __iter__(self) -> Iterator[ProximableFunctional]:
        """Iterate over the functionals."""
        return iter(self.functionals)

    def __len__(self) -> int:
        """Return the number of functionals."""
        return len(self.functionals)
