"""Get a nested attribute."""

from collections.abc import Mapping
from typing import TypeVar, cast, overload

T = TypeVar('T')


@overload
def getnestedattr(obj: object, *attrs: str, default: None = ..., return_type: None = ...) -> object | None: ...
@overload
def getnestedattr(obj: object, *attrs: str, default: T = ..., return_type: None = ...) -> T: ...
@overload
def getnestedattr(obj: object, *attrs: str, default: None = ..., return_type: type[T] = ...) -> T | None: ...
@overload
def getnestedattr(obj: object, *attrs: str, default: T = ..., return_type: type[T] = ...) -> T: ...


def getnestedattr(obj: object, *attrs: str, default: T | None = None, return_type: type[T] | None = None) -> T | None:
    """
    Get a nested attribute, or return a default if any step fails.

    Parameters
    ----------
    obj
        object to get attribute from
    attrs
        attribute names to get
    default
        value to return if any step fails
    return_type
        type to cast the result to (only for type hinting)
    """
    if return_type is not None and default is not None and not isinstance(default, return_type):
        raise TypeError('default must be of the same type as return_type')
    for attr in attrs:
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return default
    return cast(T, obj)


@overload
def getnesteditem(obj: Mapping, *items: str, default: None = ..., return_type: None = ...) -> object | None: ...
@overload
def getnesteditem(obj: Mapping, *items: str, default: T = ..., return_type: None = ...) -> T: ...
@overload
def getnesteditem(obj: Mapping, *items: str, default: None = ..., return_type: type[T] = ...) -> T | None: ...
@overload
def getnesteditem(obj: Mapping, *items: str, default: T = ..., return_type: type[T] = ...) -> T: ...


def getnesteditem(obj: Mapping, *items: str, default: T | None = None, return_type: type[T] | None = None) -> T | None:
    """
    Get a nested item, or return a default if any step fails.

    Parameters
    ----------
    obj
        object to get attribute from
    items
        item names to get
    default
        value to return if any step fails
    return_type
        type to cast the result to (only for type hinting)
    """
    if return_type is not None and default is not None and not isinstance(default, return_type):
        raise TypeError('default must be of the same type as return_type')
    for item in items:
        try:
            obj = obj[item]
        except (KeyError, TypeError):
            return default
    return cast(T, obj)
