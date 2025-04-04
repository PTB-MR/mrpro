"""Summarize the values of a tensor to a string."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class HasShortStr(Protocol):
    """Object implemtining shortstr."""

    def __shortstr__(self) -> str:
        """Return a short string representation."""


def summarize_object(obj: object) -> str:
    """Summarize object to a human readable representation.

    Parameters
    ----------
    obj
        The object to summarize.
        This method has special cases:
            For sequences of numeric values and tensors, a summary of shape and value is returned.
            If __shortstr__ is implemented, it is used.
            For torch.nn.Modules, only the name is used.
            For other objects, obj.__str__() will be used.

    Returns
    -------
        The string summary/short representation.

    """
    if isinstance(obj, torch.Tensor):
        return summarize_values(obj)
    if isinstance(obj, tuple | list):
        try:
            torch.as_tensor(obj)
        except Exception:  # noqa: BLE001
            return str(obj)
        return summarize_values(obj)
    if isinstance(obj, HasShortStr):
        return obj.__shortstr__()
    if isinstance(obj, torch.nn.Module):
        return f'{type(obj).__name__}(...)'
    return str(obj)


def summarize_values(value: torch.Tensor | Sequence[float], summarization_threshold: int = 6) -> str:
    """Summarize the values of a tensor or sequence to a string.

    Returns a string representation of the object and its values.

    Parameters
    ----------
    value
        The object to summarize.
    summarization_threshold
        The threshold of total array elements triggering summarization.

    Returns
    -------
        A string summary of the shape and values.
    """
    string = []
    if isinstance(value, torch.Tensor):
        if value.shape:
            string.append(f'Tensor<{", ".join(map(str, value.shape))}>,')
        elif value.numel():
            string.append('Tensor:')
        else:
            return 'Tensor'
    elif not len(value):
        return f'{type(value).__name__}'
    else:
        string.append(f'{type(value).__name__}<{len(value)}>,')
    value = torch.as_tensor(value, device='cpu')
    constant = False
    if value.numel() == 0:
        pass
    elif value.is_complex():
        # workaround for missing unique of complex values
        unique = torch.view_as_complex(torch.unique(torch.view_as_real(value.ravel()), dim=0))
        if len(unique) == 1:
            string.append(f'constant {unique.item():.3g}')
            constant = True
        elif value.numel() > summarization_threshold:
            magnitude = value.abs()
            string.append(
                f'|x| ∈ [{magnitude.amin().item():.3g}, {magnitude.amax().item():.3g}], μ={value.mean().item():.3g},'
            )
    else:  # real valued
        min_value, max_value = value.min(), value.max()
        if torch.isclose(min_value, max_value):
            if min_value.dtype == torch.bool:
                string.append(f'constant {min_value.item()}')
            else:
                string.append(f'constant {min_value.item():.3g}')
            constant = True
        elif value.numel() > summarization_threshold:
            mean = (value + 0.0).mean()  # +0. to force float for dtype.int64
            string.append(f'x ∈ [{min_value.item():.3g}, {max_value.item():.3g}], μ={mean.item():.3g},')
    if not constant:  # no need to add values for constant tensor
        edgeitems = 1 if summarization_threshold < 4 else 2
        with torch._tensor_str.printoptions(threshold=summarization_threshold, edgeitems=edgeitems):
            values = ''.join(
                torch._tensor_str._tensor_str(torch.as_tensor(value.ravel(), device='cpu'), 0).splitlines()
            )
            string.append(values)
    return ' '.join(string)
