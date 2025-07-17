"""Standardize an argument to a fixed-length tuple."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar('T')


def to_tuple(length: int, arg: Sequence[T] | T) -> tuple[T, ...]:
    """Standardize an argument to a fixed-length tuple.

    If the argument is a sequence, it checks if its length matches the
    specified dimension. If it's a single value, it replicates it `dim` times.

    Parameters
    ----------
    length
        The expected length of the sequence.
    arg
        The argument to check. Can be a single value of type T or a
        sequence of T.

    Returns
    -------
        A tuple of length `dim` containing elements of type T.

    Raises
    ------
    ValueError
        If `arg` is a sequence and its length does not match `length`.
    """
    if isinstance(arg, Sequence):
        if not len(arg) == length:
            raise ValueError(f'The arguments must be either single values or have length {length}. Got {arg}.')
        return tuple(arg)
    return (arg,) * length
