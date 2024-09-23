"""Base Class Functional."""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from mrpro.operators.Operator import Operator


class Functional(Operator[torch.Tensor, tuple[torch.Tensor]]):
    """Functional Base Class."""

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

    def _throw_if_negative_or_complex(self, x: torch.Tensor | float, name: str = 'sigma'):
        """Throw an exception if any element of x is negative or complex.

        Parameters
        ----------
        x
            input to be checked
        name
            name used in error message
        """
        if isinstance(x, float) and x >= 0 or isinstance(x, torch.Tensor) and not x.dtype.is_complex and (x >= 0).all():
            return
        raise ValueError(f'The parameter {name} must be real and contain only positive values')


class ProximableFunctional(Functional, ABC):
    """ProximableFunction Base Class."""

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
