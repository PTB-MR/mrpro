"""Optimizer Status base class."""

import typing

import torch


class OptimizerStatus(typing.TypedDict):
    """Base class for OptimizerStatus."""

    solution: tuple[torch.Tensor, ...]
    """Current estimate(s) of the solution. """

    iteration_number: int
    """Current iteration of the (iterative) algorithm."""
