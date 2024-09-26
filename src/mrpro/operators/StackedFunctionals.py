"""Base Class Functional."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TypeVarTuple, cast, overload

import torch

from mrpro.operators.Functional import Functional, ProximableFunctional
from mrpro.operators.Operator import Operator

Tp = TypeVarTuple('Tp')
Tp2 = TypeVarTuple('Tp2')


class StackedFunctionals(Operator[*Tp, tuple[torch.Tensor]]):
    """A class to stack multiple functionals together.

    This is a separable sum of the functionals. The forward method returns the sum of the functionals.
    """

    @overload
    def __init__(self: StackedFunctionals[torch.Tensor], f0: Functional, /): ...

    @overload
    def __init__(self: StackedFunctionals[torch.Tensor, torch.Tensor], f0: Functional, f1: Functional, /): ...

    @overload
    def __init__(
        self: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        f3: Functional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        f3: Functional,
        f4: Functional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedFunctionals[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            *tuple[torch.Tensor, ...],
        ],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        f3: Functional,
        f4: Functional,
        /,
        *f: Functional,
    ): ...

    @overload
    def __init__(self: StackedFunctionals, *funtionals: Functional): ...

    def __init__(self: StackedFunctionals[*Tp], *functionals: Functional) -> None:
        """Initialize the StackedFunctionals object.

        Parameters
        ----------
        functionals
            The functionals to be stacked.
        """
        super().__init__()
        if not len(functionals):
            raise ValueError('At least one functional is required')
        self.functionals = functionals

    def forward(self: StackedFunctionals[*Tp], *x: *Tp) -> tuple[torch.Tensor,]:
        """Apply the functionals to the inputs and return the sum of the results.

        Parameters
        ----------
        x
            The inputs to the functionals.

        Returns
        -------
        The sum of the results of the functionals.
        """
        ret = sum((f(xi)[0] for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)))
        assert isinstance(ret, torch.Tensor)  # Type hinting # noqa: S101
        return (ret,)

    def __iter__(self: StackedFunctionals[*Tp]) -> Iterator[Functional]:
        """Iterate over the functionals."""
        return iter(self.functionals)

    @overload
    def __or__(self: StackedFunctionals[*Tp], other: Functional) -> StackedFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedFunctionals[*Tp], other: StackedFunctionals[torch.Tensor]
    ) -> StackedFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedFunctionals[*Tp], other: StackedFunctionals[torch.Tensor, torch.Tensor]
    ) -> StackedFunctionals[*Tp, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedFunctionals[*Tp], other: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> StackedFunctionals[*Tp, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedFunctionals[*Tp],
        other: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]],
    ) -> StackedFunctionals[*tuple[torch.Tensor, ...]]: ...

    def __or__(self: StackedFunctionals[*Tp], other: Functional | StackedFunctionals) -> StackedFunctionals:
        """Stack functionals."""
        if isinstance(other, StackedFunctionals):
            return StackedFunctionals(*self.functionals, *other.functionals)

        elif isinstance(other, Functional):
            return StackedFunctionals(*self.functionals, other)
        else:
            raise TypeError(f'unsupported type {type(other)}')

    def __ror__(self: StackedFunctionals[*Tp], other: Functional) -> StackedFunctionals[torch.Tensor, *Tp]:
        """Stack functionals."""
        if isinstance(other, Functional):
            return cast(StackedFunctionals[torch.Tensor, *Tp], StackedFunctionals(other, *self.functionals))
        else:
            raise TypeError(f'unsupported type {type(other)}')


class StackedProximableFunctionals(StackedFunctionals[*Tp]):
    """Stacked Proximable Functionals.

    This is a separable sum of the functionals. The forward method returns the sum of the functionals.
    "
    """

    functionals: tuple[ProximableFunctional, ...]

    @overload
    def __init__(self: StackedProximableFunctionals[torch.Tensor], f0: ProximableFunctional, /): ...

    @overload
    def __init__(
        self: StackedProximableFunctionals[torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        f3: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        f3: ProximableFunctional,
        f4: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: StackedProximableFunctionals[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            *tuple[torch.Tensor, ...],
        ],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        f3: ProximableFunctional,
        f4: ProximableFunctional,
        /,
        *f: ProximableFunctional,
    ): ...

    @overload
    def __init__(self: StackedProximableFunctionals, *funtionals: ProximableFunctional): ...

    def __init__(self: StackedProximableFunctionals[*Tp], *functionals: ProximableFunctional) -> None:
        """Initialize the StackedProximableFunctionals object.

        Parameters
        ----------
        functionals
            The functionals to be stacked.
        """
        super(StackedFunctionals, self).__init__()
        self.functionals = functionals

    def prox(self: StackedProximableFunctionals[*Tp], *x: *Tp, sigma: float | torch.Tensor = 1) -> tuple[*Tp]:
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
        ret = tuple(
            f.prox(xi, sigma)[0] for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)
        )
        return cast(tuple[*Tp], ret)

    def prox_convex_conj(
        self: StackedProximableFunctionals[*Tp], *x: *Tp, sigma: float | torch.Tensor = 1
    ) -> tuple[*Tp]:
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
        ret = tuple(
            f.prox_convex_conj(xi, sigma)[0]
            for f, xi in zip(self.functionals, cast(tuple[torch.Tensor, ...], x), strict=True)
        )
        return cast(tuple[*Tp], ret)

    @overload  # type: ignore[override]
    def __or__(
        self: StackedProximableFunctionals[*Tp], other: ProximableFunctional
    ) -> StackedProximableFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp], other: StackedProximableFunctionals[torch.Tensor]
    ) -> StackedProximableFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp], other: StackedProximableFunctionals[torch.Tensor, torch.Tensor]
    ) -> StackedProximableFunctionals[*Tp, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp],
        other: StackedProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> StackedProximableFunctionals[*Tp, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals,
        other: StackedProximableFunctionals[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]
        ],
    ) -> StackedProximableFunctionals[*tuple[torch.Tensor, ...]]: ...

    @overload
    def __or__(self: StackedProximableFunctionals[*Tp], other: Functional) -> StackedFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp], other: StackedFunctionals[torch.Tensor]
    ) -> StackedFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp], other: StackedFunctionals[torch.Tensor, torch.Tensor]
    ) -> StackedFunctionals[*Tp, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp], other: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> StackedFunctionals[*Tp, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __or__(
        self: StackedProximableFunctionals[*Tp],
        other: StackedFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]],
    ) -> StackedFunctionals[*tuple[torch.Tensor, ...]]: ...

    def __or__(  # type: ignore[misc]
        self: StackedProximableFunctionals[*Tp],
        other: Functional | StackedFunctionals | ProximableFunctional | StackedProximableFunctionals,
    ) -> StackedProximableFunctionals | StackedFunctionals:
        """Stack functionals."""
        if isinstance(other, StackedProximableFunctionals):
            return StackedProximableFunctionals(*self.functionals, *other.functionals)
        if isinstance(other, ProximableFunctional):
            return StackedProximableFunctionals(*self.functionals, other)
        if isinstance(other, StackedFunctionals):
            return StackedFunctionals(*self.functionals, *other.functionals)
        if isinstance(other, Functional):
            return StackedFunctionals(*self.functionals, other)

        raise TypeError(f'unsupported type {type(other)}')

    def __ror__(
        self: StackedProximableFunctionals[*Tp], other: Functional | ProximableFunctional
    ) -> StackedProximableFunctionals[torch.Tensor, *Tp] | StackedFunctionals[torch.Tensor, *Tp]:
        """Stack functionals."""
        if isinstance(other, ProximableFunctional):
            return StackedProximableFunctionals(other, *self.functionals)
        if isinstance(other, Functional):
            return StackedFunctionals(other, *self.functionals)
        else:
            raise TypeError(f'unsupported type {type(other)}')
