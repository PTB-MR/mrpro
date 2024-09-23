"""Base Class Functional."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Self

import torch

from mrpro.operators.Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

    target: torch.Tensor
    weight: torch.Tensor
    scale: torch.Tensor

    def __init__(
        self,
        weight: torch.Tensor | complex = 1.0,
        target: torch.Tensor | None | complex = None,
        scale: torch.Tensor | float = 1.0,
        dim: int | Sequence[int] | None = None,
        divide_by_n: bool = False,
        keepdim: bool = False,
    ) -> None:
        r"""Initialize a Functional.

        We assume that functionals are given in the form
            :math:`f(x) = scale* \phi( weight ( x - target))`
        for some functional :math:`\phi`.

        Parameters
        ----------
        weight
            weighting of the norm (see above)
        target
            element to which distance is taken - often data tensor (see above)
        scale
            scaling factor for the functional, must be real and non-negative
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
        self.register_buffer('scale', torch.as_tensor(scale))
        if self.scale.dtype.is_complex or (self.scale < 0).any():
            raise ValueError('The parameter scale must be real and should not contain negative values')

        if isinstance(dim, int):
            dim = (dim,)
        elif isinstance(dim, Sequence):
            dim = tuple(dim)
        self.dim = dim
        self.divide_by_n = divide_by_n
        self.keepdim = keepdim

    def __mul__(self, other: float | torch.Tensor) -> Self:
        """Scale Functional."""
        if not isinstance(other, float | torch.Tensor):
            raise NotImplementedError
        return self.__class__(
            weight=self.weight,
            target=self.target,
            scale=self.scale * other,
            dim=self.dim,
            divide_by_n=self.divide_by_n,
            keepdim=self.keepdim,
        )

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
    """ProximableFunction Base Class."""

    @abstractmethod
    def prox(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        r"""Apply proximal operator.

        Applies :math:`prox_{f}(x) = argmin_{p} (f(p) + 1/2 \|x-p\|^{2}`.
        to a given `x`, i.e. finds `p`.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            Proximal operator applied to the input tensor.
        """

    def prox_convex_conj(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        r"""Apply proximal of convex conjugate of functional.

        Applies :math:`prox_{\sigma f*}(x) = argmin_{p} (f(p) + 1/2 \|x-p\|^{2}`,
        where f* denotes the convex conjugate of f, to a given `x`, i.e. finds `p`.

        Parameters
        ----------
        x
            input tensor

        Returns
        -------
            Proximal operator  of the convex conjugate applied to the input tensor.
        """
        return (x - self.prox(x)[0],)
