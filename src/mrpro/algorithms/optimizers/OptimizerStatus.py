"""Optimizer Status base class."""

import torch
from typing_extensions import TypedDict


class OptimizerStatus(TypedDict):
    """Base class for OptimizerStatus."""

    solution: tuple[torch.Tensor, ...]
    """Current estimate(s) of the solution. """

    iteration_number: int
    """Current iteration of the (iterative) algorithm."""
