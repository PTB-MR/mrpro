from collections.abc import Sequence
from types import EllipsisType
from typing import Self, TypeVar

import torch

from mrpro.operators import LinearOperator, Operator
from mrpro.operators.LinearOperator import LinearOperatorSum

_SingleIdxType = int | slice | EllipsisType | Sequence[int]
_IdxType = _SingleIdxType | tuple[_SingleIdxType, _SingleIdxType]

T = TypeVar('T', bound=Operator)


class ZeroOp(LinearOperator):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        return (torch.zeros_like(x),)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        return (torch.zeros_like(x),)

    def __add__(self, other: T) -> T:
        return other


class LinearOperatorMatrix(Operator):
    """Matrix of Linear Operators."""

    def __init__(self, operators: Sequence[Sequence[LinearOperator]]):
        """Initialize Linear Operator Matrix.

        Parameters
        ----------
        operators
            A sequence of rows, which are sequences of Linear Operators.
        """
        if not all(isinstance(op, LinearOperator) for row in operators for op in row):
            raise ValueError('All elements should be Linear Operators.')
        if not all(len(row) == len(operators[0]) for row in operators):
            raise ValueError('All rows should have the same length.')
        super().__init__()
        self._operators = torch.nn.ModuleList(torch.nn.ModuleList(row) for row in operators)
        self._shape = (len(operators), len(operators[0]) if operators else 0)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the Operator Matrix."""
        return self._shape

    def forward(self, *x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Apply the operator to the input.

        Parameters
        ----------
        x
            Input tensors. Requires the same number of tensors as the operator has columns.

        Returns
        -------
            Output tensors. The same number of tensors as the operator has rows.
        """
        if len(x) != self.shape[1]:
            raise ValueError('Input should have the same length as the operator has columns.')
        return tuple(sum(op(xi)[0] for op, xi in zip(row, x, strict=True)) for row in self._operators)

    def __getitem__(self, idx: _IdxType) -> Self:
        """Index the Operator Matrix.

        Parameters
        ----------
        idx
            Index or slice to select rows and columns.
        """
        idxs: tuple[_SingleIdxType, _SingleIdxType] = idx if isinstance(idx, tuple) else (idx, slice(None))
        if len(idxs) > 2:
            raise IndexError('Too many indices for LinearOperatorMatrix')

        def _to_numeric_index(idx: slice | int | Sequence[int] | EllipsisType, length: int) -> Sequence[int]:
            if isinstance(idx, slice):
                return range(*idx.indices(length))
            if isinstance(idx, int):
                return (idx,)
            if idx is Ellipsis:
                return range(length)
            if isinstance(idx, Sequence):
                return idx
            else:
                raise IndexError('Invalid index type')

        row_numbers = _to_numeric_index(idxs[0], self._shape[0])
        col_numbers = _to_numeric_index(idxs[1], self._shape[1])

        sliced_operators = [
            [row[col] for col in col_numbers] for i, row in enumerate(self._operators) if i in row_numbers
        ]
        return self.__class__(sliced_operators)

    def __repr__(self):
        """Representation of the Operator Matrix."""
        return f'LinearOperatorMatrix(shape={self._shape}, operators={self._operators})'

    # Note: The type ignores are needed because we currently cannot do arithmetic operations with non-linear operators.
    def __add__(self, other: Self | LinearOperator | torch.Tensor | complex) -> Self:  # type: ignore[override]
        operators: list[list[LinearOperator]] = []
        if isinstance(other, LinearOperatorMatrix):
            if self.shape != other.shape:
                raise ValueError('OperatorMatrix shapes do not match.')
            for self_row, other_row in zip(self._operators, other._operators, strict=False):
                operators.append([s + o for s, o in zip(self_row, other_row, strict=False)])
        elif isinstance(other, LinearOperator | torch.Tensor | complex):
            if not self.shape[0] == self.shape[1]:
                raise NotImplementedError('Cannot add a LinearOperator to a non-square OperatorMatrix.')
            for i, self_row in enumerate(self._operators):
                operators.append([op + other if i == j else op for j, op in enumerate(self_row)])
        else:
            return NotImplemented
        return self.__class__(operators)

    def __radd__(self, other: Self | LinearOperator | torch.Tensor | complex) -> Self:  # type: ignore[override]
        return self.__add__(other)

    def __mul__(self, other: torch.Tensor | Sequence[torch.Tensor]) -> Self:  # type: ignore[override]
        """LinearOperatorMatrix*Tensor multiplication."""
        if isinstance(other, torch.Tensor):
            other = (other,) * self.shape[1]
        elif len(other) != self.shape[1]:
            raise ValueError('Other should have the same length as the operator has columns.')
        operators = []
        for row in self._operators:
            operators.append([op * o for op, o in zip(row, other, strict=True)])
        return self.__class__(operators)

    def __rmul__(self, other: torch.Tensor | Sequence[torch.Tensor]) -> Self:  # type: ignore[override]
        """Tensor*LinearOperatorMatrix multiplication."""
        if isinstance(other, torch.Tensor):
            other = (other,) * self.shape[0]
        elif len(other) != self.shape[0]:
            raise ValueError('Other should have the same length as the operator has rows.')
        operators = []
        for row, o in zip(self._operators, other, strict=True):
            operators.append([o * op for op in row])
        return self.__class__(operators)

    def __matmul__(self, other: LinearOperator | Self) -> Self:  # type: ignore[override]
        """Composition of operators."""
        if isinstance(other, LinearOperator):
            return self._binary_operation(other, '__matmul__')
        elif isinstance(other, LinearOperatorMatrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError('OperatorMatrix shapes do not match.')
            new_operators = []
            for row in self._operators:
                row: list[LinearOperator]
                new_row = []
                for other_col in zip(*other._operators, strict=True):
                    other_col: list[LinearOperator]
                    elements = [s @ o for s, o in zip(row, other_col, strict=True)]
                    new_row.append(LinearOperatorSum(*elements))
                new_operators.append(new_row)
            return self.__class__(new_operators)
        return NotImplemented  # type: ignore[unreachable]

    @property
    def H(self) -> Self:
        """Adjoints of the operators."""
        return self.__class__([[op.H for op in row] for row in zip(*self._operators, strict=True)])

    @classmethod
    def from_diagonal(cls, operators: Sequence[LinearOperator]):
        """Create a diagonal Operator Matrix.

        Parameters
        ----------
        operators
            Sequence of Linear Operators to be placed on the diagonal.
        """
        operator_matrix = [
            [op if i == j else ZeroOp() for j in range(len(operators))] for i, op in enumerate(operators)
        ]
        return cls(operator_matrix)
