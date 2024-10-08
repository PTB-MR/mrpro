"""Data information dataclass."""

import dataclasses
from abc import ABC
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch
from einops import rearrange

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension

T = TypeVar('T', torch.Tensor, Rotation, SpatialDimension)


def rearrange_info_fields(field: T, pattern: str, additional_info: dict[str, int] | None = None) -> T:
    """Change the shape of the fields in DataInfo."""
    axes_lengths = {} if additional_info is None else additional_info
    if isinstance(field, Rotation):
        return Rotation.from_matrix(rearrange(field.as_matrix(), pattern, **axes_lengths))
    elif isinstance(field, SpatialDimension):
        return SpatialDimension(
            z=rearrange(field.z, pattern, **axes_lengths),
            y=rearrange(field.y, pattern, **axes_lengths),
            x=rearrange(field.x, pattern, **axes_lengths),
        )
    else:
        return rearrange(field, pattern, **axes_lengths)


@dataclass(slots=True)
class DataIdx(MoveDataMixin, ABC):
    """Data index for each data entry."""

    average: torch.Tensor
    """Signal average."""

    slice: torch.Tensor
    """Slice number (multi-slice 2D)."""

    contrast: torch.Tensor
    """Echo number in multi-echo."""

    phase: torch.Tensor
    """Cardiac phase."""

    repetition: torch.Tensor
    """Counter in repeated/dynamic acquisitions."""

    set: torch.Tensor
    """Sets of different preparation, e.g. flow encoding, diffusion weighting."""

    segment: torch.Tensor
    """Counter for segmented acquisitions."""

    user0: torch.Tensor
    """User index 0."""

    user1: torch.Tensor
    """User index 1."""

    user2: torch.Tensor
    """User index 2."""

    user3: torch.Tensor
    """User index 3."""

    user4: torch.Tensor
    """User index 4."""

    user5: torch.Tensor
    """User index 5."""

    user6: torch.Tensor
    """User index 6."""

    user7: torch.Tensor
    """User index 7."""


@dataclass(slots=True)
class DataInfo(MoveDataMixin, ABC):
    """Data information for each data entry."""

    idx: DataIdx
    """Indices describing data (e.g. slice number or repetition number)."""

    orientation: Rotation
    """Rotation describing the orientation of the readout, phase and slice encoding direction."""

    patient_table_position: SpatialDimension[torch.Tensor]
    """Offset position of the patient table, in LPS coordinates [m]."""

    physiology_time_stamp: torch.Tensor
    """Time stamps relative to physiological triggering, e.g. ECG. Not in s but in vendor-specific time units"""

    position: SpatialDimension[torch.Tensor]
    """Center of the excited volume, in LPS coordinates relative to isocenter [m]."""

    user_float: torch.Tensor
    """User-defined float parameters."""

    user_int: torch.Tensor
    """User-defined int parameters."""

    def _apply_(self, modify_data_info_field: Callable) -> None:
        """Go through all fields of DataInfo object and apply function in-place.

        Parameters
        ----------
        modify_data_info_field
            Function which takes DataInfo fields as input and returns modified DataInfo field
        """
        for field in dataclasses.fields(self):
            current = getattr(self, field.name)
            if dataclasses.is_dataclass(current):
                for subfield in dataclasses.fields(current):
                    subcurrent = getattr(current, subfield.name)
                    setattr(current, subfield.name, modify_data_info_field(subcurrent))
            else:
                setattr(self, field.name, modify_data_info_field(current))
        return None
