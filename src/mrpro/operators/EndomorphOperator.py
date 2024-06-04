"""Endomorph Operators."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec
from typing import Protocol
from typing import TypeAlias
from typing import TypeVar
from typing import TypeVarTuple
from typing import cast
from typing import overload

import torch

from mrpro.operators.Operator import Operator

Tin = TypeVarTuple('Tin')
Tout = TypeVar('Tout', bound=tuple[torch.Tensor, ...], covariant=True)
P = ParamSpec('P')
Wrapped: TypeAlias = Callable[P, Tout]
F = TypeVar('F', bound=Wrapped)


class _EndomorphCallable(Protocol):
    """A callable with the same number of tensor inputs and outputs.

    This is a protocol for a callable that takes a variadic number of tensor inputs
    and returns the same number of tensor outputs.

    This is only implemented for up to 10 inputs, if more inputs are given, the return
    will be a variadic number of tensors.

    This Protocol is used to decorate functions with the `endomorph` decorator.
    """

    @overload
    def __call__(self, /) -> tuple[()]: ...
    @overload
    def __call__(self, x1: torch.Tensor, /) -> tuple[torch.Tensor]: ...

    @overload
    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor, /
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        /,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        /,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        x9: torch.Tensor,
        /,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        x9: torch.Tensor,
        x10: torch.Tensor,
        /,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]: ...

    @overload
    def __call__(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
        x5: torch.Tensor,
        x6: torch.Tensor,
        x7: torch.Tensor,
        x8: torch.Tensor,
        x9: torch.Tensor,
        x10: torch.Tensor,
        /,
        *args: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        *tuple[torch.Tensor, ...],
    ]: ...

    @overload
    def __call__(self, /, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...

    def __call__(self, /, *args: torch.Tensor) -> tuple[torch.Tensor, ...]: ...


def endomorph(f: F, /) -> _EndomorphCallable:
    """Decorate a function to make it an endomorph callable."""
    return f


class EndomorphOperator(Operator[*tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]):
    """Endomorph Operator.

    Endomorph Operators have N  tensor inputs and exactly N outputs.
    """

    @endomorph
    def __call__(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return super().__call__(*x)

    @endomorph
    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return x

    def __matmul__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator composition."""
        return cast(Operator[*Tin, Tout], super().__matmul__(other))

    def __rmatmul__(self, other: Operator[*Tin, Tout]) -> Operator[*Tin, Tout]:
        """Operator composition."""
        return other.__matmul__(cast(Operator[*Tin, tuple[*Tin]], self))
