"""Base Class (Elemetary)(Proximable)Functional and (Proximable)StackedFunctionals."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TypeVarTuple, cast, overload

import torch

from mrpro.operators.Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

    def _throw_if_negative_or_complex(
        self, x: torch.Tensor | float, message: str = 'sigma must be real and contain only positive values'
    ):
        """Throw an exception if any element of x is negative or complex.

        Raises a ValueError if x contains negative or complex values.

        Parameters
        ----------
        x
            input to be checked
        message
            error message that is raised if x contains negative or complex values
        """
        if (isinstance(x, float | int) and x >= 0) or (
            isinstance(x, torch.Tensor) and not x.dtype.is_complex and (x >= 0).all()
        ):
            return
        raise ValueError(message)

    def __or__(self, other: Functional) -> StackedFunctionals[torch.Tensor, torch.Tensor]:
        """Create a StackedFunctionals object from two functionals.

        Parameters
        ----------
        other
            second functional to be stacked

        Returns
        -------
            StackedFunctionals object
        """
        if not isinstance(other, Functional):
            return NotImplemented  # type: ignore[unreachable]
        return StackedFunctionals(self, other)


class ElementaryFunctional(Functional):
    r"""Elementary functional base class.

    An elementary functional is a functional that can be written as
    :math:`f(x) = \phi( weight ( x - target))`, returning a real value.
    It does not require another functional for initialization.
    """

    def __init__(
        self,
        weight: torch.Tensor | complex = 1.0,
        target: torch.Tensor | None | complex = None,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize a Functional.

        We assume that functionals are given in the form
            :math:`f(x) = \phi( weight ( x - target))`
        for some functional :math:`\phi`.

        Parameters
        ----------
        weight
            weighting of the norm (see above)
        target
            element to which distance is taken - often data tensor (see above)
        dim
            dimension(s) over which norm is calculated.
            All other dimensions of  `weight ( x - target))` will be treated as batch dimensions.
        divide_by_n
            if True, the result is scaled by the number of elements of the dimensions index by `dim` in
            the tensor `weight ( x - target))`. If true, the norm is thus calculated as the mean,
            else the sum.
        keepdim
            if true, the dimension(s) of the input indexed by dim are mainted and collapsed to singeltons,
            else they are removed from the result.

        """
        super().__init__()
        self.register_buffer('weight', torch.as_tensor(weight))
        if target is None:
            target = torch.tensor(0, dtype=torch.float32)
        self.register_buffer('target', torch.as_tensor(target))
        if isinstance(dim, int):
            dim = (dim,)
        elif isinstance(dim, Sequence):
            dim = tuple(dim)
        self.dim = dim
        self.divide_by_n = divide_by_n
        self.keepdim = keepdim

    def _divide_by_n(self, x: torch.Tensor, shape: None | Sequence[int]) -> torch.Tensor:
        """Apply factor for normalization.

        Input is scaled by the number of elements of either the input
        or optional shape.

        Parameters
        ----------
        x
            input to be scaled.
        shape
            input will be divided by the product these numbers.
            If None, it divides by the number of elements of the input.

        Returns
        -------
            new scaled down tensor.
        """
        if not self.divide_by_n:
            return x
        if shape is None:
            shape = x.shape
        if self.dim is not None:
            size = [shape[i] for i in self.dim]
        else:
            size = list(shape)
        return x / math.prod(size)


class ProximableFunctional(Functional, ABC):
    r"""ProximableFunctional Base Class.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that solves the problem :math:`\min_x f(x) + 1/(2\sigma) ||x - y||^2`.
    """

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        r"""Apply proximal operator.

        Applies :math:`prox_{\sigma f}(x) = argmin_{p} (f(p) + 1/(2*sigma) \|x-p\|^{2}`.
        to a given `x`, i.e. finds `p`.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor, must be positive

        Returns
        -------
            Proximal operator applied to the input tensor.
        """

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        r"""Apply proximal of convex conjugate of functional.

        Applies :math:`prox_{\sigma f*}(x) = argmin_{p} (f(p) + 1/(2*sigma) \|x-p\|^{2}`,
        where f* denotes the convex conjugate of f, to a given `x`, i.e. finds `p`.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor, must be positive

        Returns
        -------
            Proximal operator  of the convex conjugate applied to the input tensor.
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.as_tensor(1.0 * sigma, device=self.target.device)
        self._throw_if_negative_or_complex(sigma)
        sigma[sigma < 1e-8] += 1e-6
        return (x - sigma * self.prox(x / sigma, 1 / sigma)[0],)

    @overload
    def __or__(self, other: ProximableFunctional) -> StackedProximableFunctionals[torch.Tensor, torch.Tensor]: ...
    @overload
    def __or__(self, other: Functional) -> StackedFunctionals[torch.Tensor, torch.Tensor]: ...

    def __or__(
        self, other: ProximableFunctional | Functional
    ) -> StackedProximableFunctionals[torch.Tensor, torch.Tensor] | StackedFunctionals[torch.Tensor, torch.Tensor]:
        """Create a StackedFunctionals object from two proximable functionals.

        Parameters
        ----------
        other
            second functional to be stacked

        Returns
        -------
            StackedFunctionals object
        """
        if isinstance(other, ProximableFunctional):
            return StackedProximableFunctionals(self, other)
        if isinstance(other, Functional):
            return StackedFunctionals(self, other)
        else:
            return NotImplemented  # type: ignore[unreachable]


class ElementaryProximableFunctional(ElementaryFunctional, ProximableFunctional):
    r"""Elementary proximable functional base class.

    An elementary functional is a functional that can be written as
    :math:`f(x) = \phi( weight ( x - target))`, returning a real value.
    It does not require another functional for initialization.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that solves the problem :math:`\min_x f(x) + 1/(2\sigma) ||x - y||^2`.
    """


Tp = TypeVarTuple('Tp')


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

    def __iter__(self) -> Iterator[Functional]:
        """Iterate over the functionals."""
        return iter(self.functionals)

    def __len__(self) -> int:
        """Return the number of functionals."""
        return len(self.functionals)

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
            return NotImplemented  # type: ignore[unreachable]

    def __ror__(self: StackedFunctionals[*Tp], other: Functional) -> StackedFunctionals[torch.Tensor, *Tp]:
        """Stack functionals."""
        if isinstance(other, Functional):
            return cast(StackedFunctionals[torch.Tensor, *Tp], StackedFunctionals(other, *self.functionals))
        else:
            return NotImplemented  # type: ignore[unreachable]


class StackedProximableFunctionals(StackedFunctionals[*Tp]):
    """Stacked Proximable Functionals.

    This is a separable sum of the functionals. The forward method returns the sum of the functionals.
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

        return NotImplemented  # type: ignore[unreachable]

    @overload
    def __ror__(
        self: StackedProximableFunctionals[*Tp], other: ProximableFunctional
    ) -> StackedProximableFunctionals[torch.Tensor, *Tp]: ...
    @overload
    def __ror__(
        self: StackedProximableFunctionals[*Tp], other: Functional
    ) -> StackedFunctionals[torch.Tensor, *Tp]: ...
    def __ror__(
        self: StackedProximableFunctionals[*Tp], other: Functional | ProximableFunctional
    ) -> StackedProximableFunctionals[torch.Tensor, *Tp] | StackedFunctionals[torch.Tensor, *Tp]:
        """Stack functionals."""
        if isinstance(other, ProximableFunctional):
            return StackedProximableFunctionals(other, *self.functionals)

        if isinstance(other, Functional):
            return StackedFunctionals(other, *self.functionals)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __iter__(self) -> Iterator[ProximableFunctional]:
        """Iterate over the functionals."""
        return iter(self.functionals)
