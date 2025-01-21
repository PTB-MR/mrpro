"""Dictionary Matching Operator."""

from collections.abc import Callable

import einops
import torch
from typing_extensions import Self, TypeVarTuple, Unpack

from mrpro.operators import Operator

Tin = TypeVarTuple('Tin')


class DictionaryMatchOp(Operator[torch.Tensor, tuple[*Tin]]):
    """Dictionary Matching Operator.

    This operator can be used for dictionary matching, for example in
    magnetic ressonance fingerprinting.

    At inizilization, the signal model needs to be provided.
    Afterwards `append` with different x values should be called.
    This operator than calculates for each x value the y value as returned by the signal model.

    To perform a match, use `__call__` and supply some y values. The operator will then perform
    dot product matching and return the y values that match.
    """

    def __init__(self, generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]]):
        """Initialize DictionaryMatchOp.

        Parameters
        ----------
        generating_function
            signal model that takes n inputs and returns a signal y.
        """
        super().__init__()
        self._f = generating_function
        self.x: list[torch.Tensor] = []
        self.y = torch.tensor([])

    def append(self, *x: Unpack[Tin]) -> Self:
        """Append `x` values to the dictionary.

        Parameters
        ----------
        x
            points where the signal model will be evaluated. For signal models
            with n inputs, n Tensors should be provided. Broadcasting is supported.

        Returns
        -------
        Self

        """
        (newy,) = self._f(*x)
        newy = newy / torch.linalg.norm(newy, dim=0, keepdim=True)
        newy = newy.flatten(start_dim=1)
        newx = [x.flatten() for x in torch.broadcast_tensors(*x)]
        if not self.x:
            self.x = newx
            self.y = newy
            return self
        self.x = [torch.cat(old, new) for old, new in zip(self.x, newx, strict=True)]
        self.y = torch.cat((self.y, newy))
        return

    def forward(self, input_signal: torch.Tensor) -> tuple[Unpack[Tin]]:
        """Perform dot-product matching.

        Given y values as input_signal, the tuple of x values in the dictionary
        that result in a signal with the highest dot-product similiary will be returned

        Parameters
        ----------
        input_signal
            y values, shape `(m ...)` where `m` is the return dimension of the signal model,
            for example time points.

        Returns
        -------
        match
            tuple of n tensors with shape (...)
        """
        if not self.x:
            raise KeyError('No keys in the dictionary. Please first add some x values using `append`.')
        similarity = einops.einsum(input_signal, self.y.conj(), 'm ..., m idx  -> idx ...').abs()
        idx = similarity.argmax(dim=0)
        match = tuple(x[idx] for x in self.x)
        return match
