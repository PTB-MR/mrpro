"""Summarize the values of a tensor to a string."""

import torch


def summarize_tensorvalues(tensor: torch.Tensor | None, summarization_threshold: int) -> str:
    """Summarize the values of a tensor to a string.

    Returns a string representation of the tensor values. If the tensor is None, the string 'None' is returned.

    Parameters
    ----------
    tensor
        The tensor to summarize.
    summarization_threshold
        The threshold for summarization.
    """
    if tensor is None:
        return 'None'
    with torch._tensor_str.printoptions(threshold=summarization_threshold):
        return torch._tensor_str._tensor_str(tensor, 0)
