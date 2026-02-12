"""Base Class Functional."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeAlias

import torch
from typing_extensions import TypeVarTuple, Unpack

import mr2.operators
from mr2.operators.Operator import Operator

if TYPE_CHECKING:
    T = TypeVarTuple('T')
    FunctionalType: TypeAlias = Operator[Unpack[T], tuple[torch.Tensor]]
else:  # python 3.10 runtime compatibility. typing_extension
    FunctionalType: TypeAlias = Operator


def throw_if_negative_or_complex(
    x: torch.Tensor | complex, message: str = 'sigma must be real and contain only positive values'
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


class ElementaryFunctional(Operator[torch.Tensor, tuple[torch.Tensor]], ABC):
    r"""Elementary functional base class.

    Here, an 'elementary' functional is a functional that can be written as
    :math:`f(x) = \phi ( \mathrm{weight} ( x - \mathrm{target}))`, returning a real value.
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
        :math:`f(x) = \phi ( \mathrm{weight} ( x - \mathrm{target}))`
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
            if true, the dimension(s) of the input indexed by `dim` are maintained and collapsed to singletons,
            else they are removed from the result.

        """
        super().__init__()
        self.weight = torch.as_tensor(weight)
        if target is None:
            target = torch.tensor(0, dtype=torch.float32)
        self.target = torch.as_tensor(target)
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


class ProximableFunctional(Operator[torch.Tensor, tuple[torch.Tensor]], ABC):
    r"""ProximableFunctional Base Class.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that yields :math:`\mathrm{argmin}_x \sigma f(x) + 1/2 ||x - y||_2^2`
    and a prox_convex_conjugate, yielding the prox of the convex conjugate.
    """

    @abstractmethod
    def prox(self, x: torch.Tensor, sigma: torch.Tensor | float = 1.0) -> tuple[torch.Tensor]:
        r"""Apply proximal operator.

        Yields :math:`\mathrm{prox}_{\sigma f}(x) = \mathrm{argmin}_{p} \sigma f(p) + 1/2 \|x-p\|_2^2` given :math:`x`
        and :math:`\sigma`.

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

        Yields :math:`\mathrm{prox}_{\sigma f^*}(x) = \mathrm{argmin}_{p} \sigma f^*(p) + 1/2 \|x-p\|_2^2`,
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
            sigma = torch.as_tensor(1.0 * sigma)
        throw_if_negative_or_complex(sigma)
        sigma = sigma.clamp(min=1e-8)
        return (x - sigma * self.prox(x / sigma, 1 / sigma)[0],)

    def __rmul__(self, scalar: torch.Tensor | complex) -> ProximableFunctional:
        """Multiply functional with scalar."""
        if not isinstance(scalar, int | float | torch.Tensor):
            return NotImplemented
        return ScaledProximableFunctional(self, scalar)

    def __or__(
        self, other: ProximableFunctional
    ) -> mr2.operators.ProximableFunctionalSeparableSum[torch.Tensor, torch.Tensor]:
        """Create a ProximableFunctionalSeparableSum from two proximable functionals.

        ``f | g`` is a separable sum with ``(f|g)(x,y) == f(x) + g(y)``.

        Parameters
        ----------
        other
            second functional to be summed

        Returns
        -------
            ProximableFunctionalSeparableSum object
        """
        if isinstance(other, ProximableFunctional):
            return mr2.operators.ProximableFunctionalSeparableSum(self, other)
        return NotImplemented


class ElementaryProximableFunctional(ElementaryFunctional, ProximableFunctional):
    r"""Elementary proximable functional base class.

    Here, an 'elementary' functional is a functional that can be written as
    :math:`f(x) = \phi ( \mathrm{weight} ( x - \mathrm{target}))`, returning a real value.
    It does not require another functional for initialization.

    A proximable functional is a functional :math:`f(x)` that has a prox implementation,
    i.e. a function that yields :math:`\mathrm{argmin}_x \sigma f(x) + 1/2 \|x - y\|^2`.
    """


class ScaledProximableFunctional(ProximableFunctional):
    """Proximable Functional scaled by a scalar."""

    def __init__(self, functional: ProximableFunctional, scale: torch.Tensor | float) -> None:
        r"""Initialize a scaled proximable functional.

        A scaled functional is a functional that is scaled by a scalar factor :math:`\alpha`,
        i.e. :math:`f(x) = \alpha g(x)`.

        Parameters
        ----------
        functional
            proximable functional to be scaled
        scale
            scaling factor, must be real and positive
        """
        super().__init__()
        self.functional = functional
        self.scale = torch.as_tensor(scale)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the functional.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            scaled output of the functional
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply the functional.

        .. note::
            Prefer calling the instance of the ScaledProximableFunctional as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return (self.scale * self.functional(x)[0],)

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
        throw_if_negative_or_complex(
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
        throw_if_negative_or_complex(
            self.scale, 'For prox_convex_conj to be defined, the scaling factor must be real and non-negative'
        )
        return (self.scale * self.functional.prox_convex_conj(x / self.scale, sigma / self.scale)[0],)
