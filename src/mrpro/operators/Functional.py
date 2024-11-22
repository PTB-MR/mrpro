"""Base Class Functional."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

import mrpro.operators
from mrpro.operators.Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

    def __rmul__(self, scalar: torch.Tensor | complex) -> Functional:
        """Multiply functional with scalar."""
        if not isinstance(scalar, int | float | torch.Tensor):
            return NotImplemented
        return ScaledFunctional(self, scalar)

    def _throw_if_negative_or_complex(
        self, x: torch.Tensor | complex, message: str = 'sigma must be real and contain only positive values'
    ) -> None:
        """Throw an ValueError if any element of x is negative or complex.

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


class ElementaryFunctional(Functional):
    r"""Elementary functional base class.

    Here, an 'elementary' functional is a functional that can be written as
    :math:`f(x) = \phi ( weight ( x - target))`, returning a real value.
    It does not require another functional for initialization.
    """

    def __init__(
        self,
        target: torch.Tensor | None | complex = None,
        weight: torch.Tensor | complex = 1.0,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize a Functional.

        We assume that functionals are given in the form
        :math:`f(x) = \phi ( weight ( x - target))`
        for some functional :math:`\phi`.

        Parameters
        ----------
        target
            target element - often data tensor (see above)
        weight
            weight parameter (see above)
        dim
            dimension(s) over which functional is reduced.
            All other dimensions of  `weight ( x - target)` will be treated as batch dimensions.
        divide_by_n
            if true, the result is scaled by the number of elements of the dimensions index by `dim` in
            the tensor `weight ( x - target)`. If true, the functional is thus calculated as the mean,
            else the sum.
        keepdim
            if true, the dimension(s) of the input indexed by dim are maintained and collapsed to singeltons,
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
        or product of the optional shape entries

        Parameters
        ----------
        x
            input to be scaled.
        shape
            input will be divided by the product of these numbers.
            If None, it will be divided by the number of elements of the input.

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
    i.e. a function that yields :math:`argmin_x \sigma f(x) + 1/2 ||x - y||^2`
    and a prox_convex_conjugate, yielding the prox of the convex conjugate.
    """

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        r"""Apply proximal operator.

        Yields :math:`prox_{\sigma f}(x) = argmin_{p} (\sigma f(p) + 1/2 \|x-p\|^{2}` given :math:`x` and :math:`\sigma`

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor, must be positive

        Returns
        -------
            Proximal operator applied to the input tensor
        """

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        r"""Apply proximal operator of convex conjugate of functional.

        Yields :math:`prox_{\sigma f^*}(x) = argmin_{p} (\sigma f^*(p) + 1/2 \|x-p\|^{2}`,
        where :math:`f^*` denotes the convex conjugate of :math:`f`, given :math:`x` and :math:`\sigma`.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor, must be positive

        Returns
        -------
            Proximal operator  of the convex conjugate applied to the input tensor
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.as_tensor(1.0 * sigma, device=self.target.device)
        self._throw_if_negative_or_complex(sigma)
        sigma[sigma < 1e-8] += 1e-6
        return (x - sigma * self.prox(x / sigma, 1 / sigma)[0],)

    def __rmul__(self, scalar: torch.Tensor | complex) -> ProximableFunctional:
        """Multiply functional with scalar."""
        if not isinstance(scalar, int | float | torch.Tensor):
            return NotImplemented
        return ScaledProximableFunctional(self, scalar)

    def __or__(self, other: ProximableFunctional) -> mrpro.operators.ProximableFunctionalSeparableSum:
        """Create a ProximableFunctionalSeparableSum object from two proximable functionals.

        Parameters
        ----------
        other
            second functional to be summed

        Returns
        -------
            ProximableFunctionalSeparableSum object
        """
        if isinstance(other, ProximableFunctional):
            return mrpro.operators.ProximableFunctionalSeparableSum(self, other)
        return NotImplemented  # type: ignore[unreachable]


class ElementaryProximableFunctional(ElementaryFunctional, ProximableFunctional):
    r"""Elementary proximable functional base class.

    Here, an *elementary* functional is a functional that can be written as
    :math:`f(x) = \phi ( weight ( x - target))`, returning a real value.
    It does not require another functional for initialization.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that yields :math:`argmin_x \sigma f(x) + 1/2 \|x - y\|^2`.
    """


class ScaledFunctional(Functional):
    """Functional scaled by a scalar."""

    def __init__(self, functional: Functional, scale: torch.Tensor | float) -> None:
        r"""Initialize a scaled functional.

        A scaled functional is a functional that is scaled by a scalar factor :math:`\alpha`,
        i.e. :math:`f(x) = \alpha g(x)`.

        Parameters
        ----------
        functional
            functional to be scaled
        scale
            scaling factor, must be real and positive
        """
        super().__init__()
        self.functional = functional
        self.register_buffer('scale', torch.as_tensor(scale))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Forward method.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            scaled output of the functional
        """
        return (self.scale * self.functional(x)[0],)


class ScaledProximableFunctional(ScaledFunctional, ProximableFunctional):
    """Proximable Functional scaled by a scalar."""

    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal Mapping.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal mapping applied to the input tensor
        """
        self._throw_if_negative_or_complex(
            self.scale, 'For prox to be defined, the scaling factor must be real and non-negative'
        )
        return (self.functional.prox(x, sigma * self.scale)[0],)

    def prox_convex_conj(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        """Proximal Mapping of the convex conjugate.

        Parameters
        ----------
        x
            input tensor
        sigma
            scaling factor

        Returns
        -------
            Proximal mapping of the convex conjugate applied to the input tensor
        """
        self._throw_if_negative_or_complex(
            self.scale, 'For prox_convex_conj to be defined, the scaling factor must be real and non-negative'
        )
        return (self.scale * self.functional.prox_convex_conj(x / self.scale, sigma / self.scale)[0],)
