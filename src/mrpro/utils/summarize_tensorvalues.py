"""Summarize the values of a tensor to a string."""

import torch


def summarize_tensorvalues(tensor: torch.Tensor | None, summarization_threshold: int = 7) -> str:
    """Summarize the values of a tensor to a string.

    Returns a string representation of the tensor values. If the tensor is None, the string 'None' is returned.

    Parameters
    ----------
    tensor
        The tensor to summarize.
    summarization_threshold
        The threshold of total array elements triggering summarization.
    edgeitems
        Number of elements at the beginning and end of each dimension to show.
    """
    if tensor is None:
        return 'None'
    if summarization_threshold < 4:
        edgeitems = 1
    elif summarization_threshold < 7:
        edgeitems = 2
    else:
        edgeitems = 3
    with torch._tensor_str.printoptions(threshold=summarization_threshold, edgeitems=edgeitems):
        return torch._tensor_str._tensor_str(tensor, 0)
