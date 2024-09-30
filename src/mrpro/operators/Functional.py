"""Base Class (Elemetary)(Proximable)Functional and (Proximable)SeparableSumFunctionals."""

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

    def __and__(self, other: Functional) -> SeparableSumFunctionals[torch.Tensor, torch.Tensor]:
        """Create a SeparableSumFunctionals object from two functionals.

        Parameters
        ----------
        other
            second functional to be summed

        Returns
        -------
            SeparableSumFunctionals object of the two functionals
        """
        if not isinstance(other, Functional):
            return NotImplemented  # type: ignore[unreachable]
        return SeparableSumFunctionals(self, other)


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
    def __and__(self, other: ProximableFunctional) -> SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor]: ...
    @overload
    def __and__(self, other: Functional) -> SeparableSumFunctionals[torch.Tensor, torch.Tensor]: ...

    def __and__(
        self, other: ProximableFunctional | Functional
    ) -> (
        SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor]
        | SeparableSumFunctionals[torch.Tensor, torch.Tensor]
    ):
        """Create a SeparableSumFunctionals object from two proximable functionals.

        Parameters
        ----------
        other
            second functional to be summed

        Returns
        -------
            SeparableSumFunctionals object of the two functionals
        """
        if isinstance(other, ProximableFunctional):
            return SeparableSumProximableFunctionals(self, other)
        if isinstance(other, Functional):
            return SeparableSumFunctionals(self, other)
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


class SeparableSumFunctionals(Operator[*Tp, tuple[torch.Tensor]]):
    """A class to sum multiple functionals together.

    This is a separable sum of the functionals, i.e., the functionals are applied to the inputs
    and the results are summed.

    `SeparableSumFunctionals(f0, f1, f2, ...)(x0, x1, x2, ...) = f0(x0) + f1(x1) + f2(x2) + ...`
    """

    @overload
    def __init__(self: SeparableSumFunctionals[torch.Tensor], f0: Functional, /): ...

    @overload
    def __init__(self: SeparableSumFunctionals[torch.Tensor, torch.Tensor], f0: Functional, f1: Functional, /): ...

    @overload
    def __init__(
        self: SeparableSumFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        /,
    ): ...

    @overload
    def __init__(
        self: SeparableSumFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        f3: Functional,
        /,
    ): ...

    @overload
    def __init__(
        self: SeparableSumFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: Functional,
        f1: Functional,
        f2: Functional,
        f3: Functional,
        f4: Functional,
        /,
    ): ...

    @overload
    def __init__(self: SeparableSumFunctionals, *funtionals: Functional): ...

    def __init__(self: SeparableSumFunctionals[*Tp], *functionals: Functional) -> None:
        """Initialize the SeparableSumFunctionals object.

        Parameters
        ----------
        functionals
            The functionals to be summed.
        """
        super().__init__()
        if not len(functionals):
            raise ValueError('At least one functional is required')
        self.functionals = functionals

    def forward(self: SeparableSumFunctionals[*Tp], *x: *Tp) -> tuple[torch.Tensor,]:
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
    def __and__(
        self: SeparableSumFunctionals[*Tp], other: Functional
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumFunctionals[*Tp], other: SeparableSumFunctionals[torch.Tensor]
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumFunctionals[*Tp], other: SeparableSumFunctionals[torch.Tensor, torch.Tensor]
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumFunctionals[*Tp], other: SeparableSumFunctionals[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumFunctionals[*Tp],
        other: SeparableSumFunctionals[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]
        ],
    ) -> SeparableSumFunctionals[*tuple[torch.Tensor, ...]]: ...

    def __and__(
        self: SeparableSumFunctionals[*Tp], other: Functional | SeparableSumFunctionals
    ) -> SeparableSumFunctionals:
        """Sum separable functionals."""
        if isinstance(other, SeparableSumFunctionals):
            return SeparableSumFunctionals(*self.functionals, *other.functionals)

        elif isinstance(other, Functional):
            return SeparableSumFunctionals(*self.functionals, other)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __rand__(self: SeparableSumFunctionals[*Tp], other: Functional) -> SeparableSumFunctionals[torch.Tensor, *Tp]:
        """Sum separable functionals."""
        if isinstance(other, Functional):
            return cast(SeparableSumFunctionals[torch.Tensor, *Tp], SeparableSumFunctionals(other, *self.functionals))
        else:
            return NotImplemented  # type: ignore[unreachable]


class SeparableSumProximableFunctionals(SeparableSumFunctionals[*Tp]):
    """Separable Sum of Proximable Functionals.

    This is a separable sum of the proximable functionals, i.e., the functionals are applied to the inputs
    and the results are summed.

    `SeparableSumProximableFunctionals(f0, f1, f2, ...)(x0, x1, x2, ...) = f0(x0) + f1(x1) + f2(x2) + ...`
    """

    functionals: tuple[ProximableFunctional, ...]

    @overload
    def __init__(self: SeparableSumProximableFunctionals[torch.Tensor], f0: ProximableFunctional, /): ...

    @overload
    def __init__(
        self: SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        f3: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(
        self: SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        f0: ProximableFunctional,
        f1: ProximableFunctional,
        f2: ProximableFunctional,
        f3: ProximableFunctional,
        f4: ProximableFunctional,
        /,
    ): ...

    @overload
    def __init__(self: SeparableSumProximableFunctionals, *funtionals: ProximableFunctional): ...

    def __init__(self: SeparableSumProximableFunctionals[*Tp], *functionals: ProximableFunctional) -> None:
        """Initialize the SeparableSumProximableFunctionals object.

        Parameters
        ----------
        functionals
            The functionals to be summed.
        """
        super(SeparableSumFunctionals, self).__init__()
        self.functionals = functionals

    def prox(self: SeparableSumProximableFunctionals[*Tp], *x: *Tp, sigma: float | torch.Tensor = 1) -> tuple[*Tp]:
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
        self: SeparableSumProximableFunctionals[*Tp], *x: *Tp, sigma: float | torch.Tensor = 1
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
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp], other: ProximableFunctional
    ) -> SeparableSumProximableFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp], other: SeparableSumProximableFunctionals[torch.Tensor]
    ) -> SeparableSumProximableFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp],
        other: SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor],
    ) -> SeparableSumProximableFunctionals[*Tp, torch.Tensor, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp],
        other: SeparableSumProximableFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> SeparableSumProximableFunctionals[*Tp, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals,
        other: SeparableSumProximableFunctionals[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]
        ],
    ) -> SeparableSumProximableFunctionals[*tuple[torch.Tensor, ...]]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp], other: Functional
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp], other: SeparableSumFunctionals[torch.Tensor]
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp], other: SeparableSumFunctionals[torch.Tensor, torch.Tensor]
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp],
        other: SeparableSumFunctionals[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> SeparableSumFunctionals[*Tp, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __and__(
        self: SeparableSumProximableFunctionals[*Tp],
        other: SeparableSumFunctionals[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, *tuple[torch.Tensor, ...]
        ],
    ) -> SeparableSumFunctionals[*tuple[torch.Tensor, ...]]: ...

    def __and__(  # type: ignore[misc]
        self: SeparableSumProximableFunctionals[*Tp],
        other: Functional | SeparableSumFunctionals | ProximableFunctional | SeparableSumProximableFunctionals,
    ) -> SeparableSumProximableFunctionals | SeparableSumFunctionals:
        """Sum separable functionals."""
        if isinstance(other, SeparableSumProximableFunctionals):
            return SeparableSumProximableFunctionals(*self.functionals, *other.functionals)
        if isinstance(other, ProximableFunctional):
            return SeparableSumProximableFunctionals(*self.functionals, other)
        if isinstance(other, SeparableSumFunctionals):
            return SeparableSumFunctionals(*self.functionals, *other.functionals)
        if isinstance(other, Functional):
            return SeparableSumFunctionals(*self.functionals, other)

        return NotImplemented  # type: ignore[unreachable]

    @overload
    def __rand__(
        self: SeparableSumProximableFunctionals[*Tp], other: ProximableFunctional
    ) -> SeparableSumProximableFunctionals[torch.Tensor, *Tp]: ...
    @overload
    def __rand__(
        self: SeparableSumProximableFunctionals[*Tp], other: Functional
    ) -> SeparableSumFunctionals[torch.Tensor, *Tp]: ...
    def __rand__(
        self: SeparableSumProximableFunctionals[*Tp], other: Functional | ProximableFunctional
    ) -> SeparableSumProximableFunctionals[torch.Tensor, *Tp] | SeparableSumFunctionals[torch.Tensor, *Tp]:
        """Sum separable functionals."""
        if isinstance(other, ProximableFunctional):
            return SeparableSumProximableFunctionals(other, *self.functionals)

        if isinstance(other, Functional):
            return SeparableSumFunctionals(other, *self.functionals)
        else:
            return NotImplemented  # type: ignore[unreachable]

    def __iter__(self) -> Iterator[ProximableFunctional]:
        """Iterate over the functionals."""
        return iter(self.functionals)
