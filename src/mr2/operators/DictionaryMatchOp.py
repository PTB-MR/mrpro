"""Dictionary Matching Operator."""

from collections.abc import Callable
from typing import cast

import einops
import torch
from typing_extensions import Self, TypeVarTuple, Unpack

from mr2.operators.Operator import Operator

Tin = TypeVarTuple('Tin')


class DictionaryMatchOp(Operator[torch.Tensor, tuple[Unpack[Tin]]]):
    r"""Dictionary Matching Operator.

    This operator can be used for dictionary matching, for example in
    magnetic ressonance fingerprinting.

    It performs absolute normalized dot product matching between a dictionary of signals,
    i.e. find the entry :math:`d^*` in the dictionary maximizing
    :math:`\left|\frac{d}{\|d\|} \cdot \frac{y}{\|y\|}\right|` and returns the
    associated signal model parameters :math:`x` generating the matching signal :math:`d^*=d(x)`.

    At initialization, a signal model needs to be provided.
    Afterwards `append` with different `x` values should be called to add entries to the dictionary.
    This operator then calculates for each `x` value the signal returned by the model.
    To perform a match, use `__call__` and supply some `y` values. The operator will then perform the
    dot product matching and return the associated `x` values.
    """

    def __init__(
        self,
        generating_function: Callable[[Unpack[Tin]], tuple[torch.Tensor,]],
        index_of_scaling_parameter: int | None = None,
    ):
        """Initialize DictionaryMatchOp.

        Parameters
        ----------
        generating_function
            signal model that takes n inputs and returns a signal y.
        index_of_scaling_parameter
            Normalized dot product matching is insensitive to overall signal scaling.
            A scaling factor (e.g. the equilibrium magnetization `m0` in `~mr2.operators.models.InversionRecovery`)
            is calculated after the dictionary matching if `index_of_scaling_parameter` is not `None`.
            `index_of_scaling_parameter` should set to the index of the scaling parameter in the signal model.

            Example:
                For ~mr2.operators.models.InversionRecovery the parameters are ``[m0, t1]`` and therefore
                `index_of_scaling_parameter` should be set to 0. The operator will then return `t1` estimated
                via dictionary matching and `m0` via a post-processing step.
                If `index_of_scaling_parameter` is None, the value returned for `m0` will be meaningless.
        """
        super().__init__()
        self._f = generating_function
        self.x: list[torch.Tensor] = []
        self.y = torch.tensor([])
        self._index_of_scaling_parameter = index_of_scaling_parameter
        self.inverse_norm_y = None if index_of_scaling_parameter is None else torch.tensor([])

    def append(self, *x: Unpack[Tin]) -> Self:
        """Append `x` values to the dictionary.

        Parameters
        ----------
        x
            points where the signal model will be evaluated. For signal models
            with n inputs, n tensors should be provided. Broadcasting is supported.

        Returns
        -------
            Self

        """
        if self._index_of_scaling_parameter is not None:
            scaling_position = self._index_of_scaling_parameter % len(x)
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

    def __call__(self, input_signal: torch.Tensor) -> tuple[Unpack[Tin]]:
        """Perform dot-product matching.

        Given an input signal (or batch of signals), this method finds the entry in the
        precomputed dictionary that has the highest absolute normalized dot product similarity.
        It then returns the set of parameters `x` that generated this dictionary entry.

        If `index_of_scaling_parameter` was set during initialization, the corresponding
        scaling parameter is estimated and inserted into the returned tuple of parameters.

        Parameters
        ----------
        input_signal
            Input signal(s) to match against the dictionary.
            Expected shape is `(m, ...)`, where `m` is the signal dimension
            (e.g., number of time points) and `(...)` are batch dimensions.

        Returns
        -------
            A tuple of tensors representing the parameters `x` from the dictionary
            that best matched the input signal(s). Each tensor in the tuple corresponds
            to a parameter, and their shapes will match the batch dimensions of `input_signal`.
        """
        return super().__call__(input_signal)

    def forward(self, input_signal: torch.Tensor) -> tuple[Unpack[Tin]]:
        """Apply forward of DictionaryMatchOp.

        .. note::
            Prefer calling the instance of the DictionaryMatchOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        if not self.x:
            raise KeyError('No keys in the dictionary. Please first add some x values using `append`.')

        # This avoids unnecessary copies mixed domain cases
        similarity = einops.einsum(input_signal.real, self.y.real, 'm ..., m idx  -> idx ...').square()
        if self.y.is_complex():
            similarity += einops.einsum(input_signal.real, self.y.imag, 'm ..., m idx  -> idx ...').square()
        if input_signal.is_complex():
            similarity += einops.einsum(input_signal.imag, self.y.real, 'm ..., m idx  -> idx ...').square()
        if self.y.is_complex() and input_signal.is_complex():
            similarity += einops.einsum(input_signal.imag, self.y.imag, 'm ..., m idx  -> idx ...').square()

        idx = similarity.argmax(dim=0)
        match = [x[idx] for x in self.x]

        if self._index_of_scaling_parameter is not None and self.inverse_norm_y is not None:
            # replace the scaling argument with the correct scaling factor
            scale = (self.y[:, idx].conj() * input_signal).sum(0) * self.inverse_norm_y[idx]
            match.insert(self._index_of_scaling_parameter, scale)

        return cast(tuple[Unpack[Tin]], tuple(match))
