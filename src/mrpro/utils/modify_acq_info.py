"""Modify AcqInfo."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from mrpro.data.AcqInfo import AcqInfo


def modify_acq_info(fun_modify: Callable, acq_info: AcqInfo) -> AcqInfo:
    """Go through all fields of AcqInfo object and apply changes.

    Parameters
    ----------
    fun_modify
        Function which takes AcqInfo fields as input and returns modified AcqInfo field
    acq_info
        AcqInfo object
    """
    # Apply function to all fields of acq_info
    for field in dataclasses.fields(acq_info):
        current = getattr(acq_info, field.name)
        if isinstance(current, torch.Tensor):
            setattr(acq_info, field.name, fun_modify(current))
        elif dataclasses.is_dataclass(current):
            for subfield in dataclasses.fields(current):
                subcurrent = getattr(current, subfield.name)
                setattr(current, subfield.name, fun_modify(subcurrent))

    return acq_info
