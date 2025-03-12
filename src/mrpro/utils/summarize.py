"""Summarize the values of a tensor to a string."""

from collections.abc import Sequence

import torch


def summarize_object(obj: object) -> str:
    """Summarize object to a human readable representation.

    Parameters
    ----------
    obj
        The object to summarize.
        This method has special cases:
            For sequences of numeric values and tensors, a summary of shape and value is returned.
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
    if isinstance(obj, torch.nn.Module):
        return f'{type(obj).__name__}(...)'
    return str(obj)


def summarize_values(value: torch.Tensor | Sequence[float], summarization_threshold: int = 4) -> str:
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
        string.append(f'Tensor{list(value.shape)!s}:')
    else:
        string.append(f'{type(value).__name__}[{len(value)}]:')
    value = torch.as_tensor(value, device='cpu')
    if value.numel() >= summarization_threshold:
        if value.is_complex():
            unique = value.unique()
            if len(unique) == 1:
                string.append(f'constant {unique}')
            else:
                magnitude = value.abs()
                string.append(f'|x| ∈ [{magnitude.amin()}, {magnitude.amax()}]')
        else:
            min_value, max_value = value.amin(), value.amax()
            if torch.isclose(min_value, max_value):
                string.append(f'constant {min_value}')
            else:
                string.append(f'x∈[{min_value}, {max_value}],')
                edgeitems = 1 if summarization_threshold < 4 else 2
                with torch._tensor_str.printoptions(threshold=summarization_threshold, edgeitems=edgeitems):
                    values = ''.join(
                        torch._tensor_str._tensor_str(torch.as_tensor(value, device='cpu'), 0).splitlines()
                    )
                    string.append(values)
    return ' '.join(string)
