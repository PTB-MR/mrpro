"""Dictionary Matching Operator."""

from collections.abc import Callable
from typing import cast

import einops
import torch
from typing_extensions import Self, TypeVarTuple, Unpack

from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')


class DictionaryMatchOp(Operator[torch.Tensor, tuple[*Tin]]):
    r"""Dictionary Matching Operator.

    This operator can be used for dictionary matching, for example in
    magnetic ressonance fingerprinting.

    It performs absolute normalized dot product matching between a dictionary of signals,
    i.e. find the entry :math:`d^*` in the dictionary maximizing
    :math:`\left|\frac{d}{\|d\|} \cdot \frac{y}{\|y\|}\right|` and returns the
    associated signal model parameters :math:`x` generating the matching signal :math:`d^*=d(x)`.

    At initialization, a signal model needs to be provided.
    Afterwards `append` with different `x` values should be called to add entries to the dictionary.
    This operator than calculates for each `x` value the signal returned by the model.
    To perform a match, use `__call__` and supply some `y` values. The operator will then perform the
    dot product matching and return the associated `x` values.
    """

    def __init__(
        self, generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]], scaling_argument: int | None = None
    ):
        """Initialize DictionaryMatchOp.

        Parameters
        ----------
        generating_function
            signal model that takes n inputs and returns a signal y.
        scaling_argument
            If one of the inputs of the signal model is purely a scaling factor, the index of this input should
            be set. This causes values provided for the scaling argument in the `append` method to be ignored
            and when performing the match, the correct scaling factor will be calculated.
            Otherwise, for pure scaling factors the values are meaningless due to the normalization.
        """
        super().__init__()
        self._f = generating_function
        self.x: list[torch.Tensor] = []
        self.y = torch.tensor([])
        self._scaling_argument = scaling_argument
        self.inverse_norm_y = None if scaling_argument is None else torch.tensor([])

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
        if self._scaling_argument is not None:
            scaling_position = self._scaling_argument % len(x)
            # replace the scaling argument with 1 in call
            (y,) = self._f(*x[:scaling_position], torch.tensor(1), *x[scaling_position + 1 :])  # type: ignore[call-arg]
            # but drop it in the dictionary
            x_list = [x.flatten() for x in torch.broadcast_tensors(*x[:scaling_position], *x[scaling_position + 1 :])]
        else:
            (y,) = self._f(*x)
            x_list = [x.flatten() for x in torch.broadcast_tensors(*x)]

        y = y.flatten(start_dim=1)
        inverse_norm_y = torch.linalg.norm(y, dim=0).reciprocal()
        y = y * inverse_norm_y

        if not self.x:
            self.x = x_list
            self.y = y
            if self.inverse_norm_y is not None:
                self.inverse_norm_y = inverse_norm_y
            return self

        self.x = [torch.cat((old, new)) for old, new in zip(self.x, x_list, strict=True)]
        self.y = torch.cat((self.y, y), dim=-1)
        if self.inverse_norm_y is not None:
            self.inverse_norm_y = torch.cat((self.inverse_norm_y, inverse_norm_y))
        return self

    def forward(self, input_signal: torch.Tensor) -> tuple[Unpack[Tin]]:
        """Perform dot-product matching.

        Given y values as input_signal, the tuple of x values in the dictionary
        that result in a signal with the highest dot-product similarity will be returned

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

        # This avoids unnessary copies mixed domain cases
        similarity = einops.einsum(input_signal.real, self.y.real, 'm ..., m idx  -> idx ...').square()
        if self.y.is_complex():
            similarity += einops.einsum(input_signal.real, self.y.imag, 'm ..., m idx  -> idx ...').square()
        if input_signal.is_complex():
            similarity += einops.einsum(input_signal.imag, self.y.real, 'm ..., m idx  -> idx ...').square()
        if self.y.is_complex() and input_signal.is_complex():
            similarity += einops.einsum(input_signal.imag, self.y.imag, 'm ..., m idx  -> idx ...').square()

        idx = similarity.argmax(dim=0)
        match = [x[idx] for x in self.x]

        if self._scaling_argument is not None and self.inverse_norm_y is not None:
            # replace the scaling argument with the correct scaling factor
            scale = (self.y[:, idx].conj() * input_signal).sum(0) * self.inverse_norm_y[idx]
            match.insert(self._scaling_argument, scale)

        return cast(tuple[Unpack[Tin]], tuple(match))
