"""Encoding limits dataclass."""

import dataclasses
from dataclasses import dataclass

from ismrmrd.xsd.ismrmrdschema.ismrmrd import encodingLimitsType, limitType
from typing_extensions import Self


@dataclass(slots=True)
class Limits:
    """Limits dataclass with min, max, and center attributes."""

    min: int = 0
    """Lower boundary."""

    max: int = 0
    """Upper boundary."""

    center: int = 0
    """Center."""

    @classmethod
    def from_ismrmrd(cls, limit_type: limitType) -> Self:
        """Create Limits from ismrmrd.limitType."""
        if limit_type is None:
            return cls()
        return cls(*dataclasses.astuple(limit_type))

    @property
    def length(self) -> int:
        """Length of the limits."""
        return self.max - self.min + 1


@dataclass(slots=True)
class EncodingLimits:
    """Encoding limits dataclass with limits for each attribute [INA2016]_.

    References
    ----------
    .. [INA2016] Inati S, Hansen M (2016) ISMRM Raw data format: A proposed standard for MRI raw datasets. MRM 77(1)
        https://doi.org/10.1002/mrm.26089

    """

    k0: Limits = dataclasses.field(default_factory=Limits)
    """First k-space encoding."""

    k1: Limits = dataclasses.field(default_factory=Limits)
    """Second k-space encoding."""

    k2: Limits = dataclasses.field(default_factory=Limits)
    """Third k-space encoding."""

    average: Limits = dataclasses.field(default_factory=Limits)
    """Signal average."""

    slice: Limits = dataclasses.field(default_factory=Limits)
    """Slice number (multi-slice 2D)."""

    contrast: Limits = dataclasses.field(default_factory=Limits)
    """Echo number in multi-echo."""

    phase: Limits = dataclasses.field(default_factory=Limits)
    """Cardiac phase."""

    repetition: Limits = dataclasses.field(default_factory=Limits)
    """Repeated/dynamic acquisitions."""

    set: Limits = dataclasses.field(default_factory=Limits)
    """Sets of different preparation."""

    segment: Limits = dataclasses.field(default_factory=Limits)
    """Segments of segmented acquisition."""

    user_0: Limits = dataclasses.field(default_factory=Limits)
    """User index 0."""

    user_1: Limits = dataclasses.field(default_factory=Limits)
    """User index 1."""

    user_2: Limits = dataclasses.field(default_factory=Limits)
    """User index 2."""

    user_3: Limits = dataclasses.field(default_factory=Limits)
    """User index 3."""

    user_4: Limits = dataclasses.field(default_factory=Limits)
    """User index 4."""

    user_5: Limits = dataclasses.field(default_factory=Limits)
    """User index 5."""

    user_6: Limits = dataclasses.field(default_factory=Limits)
    """User index 6."""

    user_7: Limits = dataclasses.field(default_factory=Limits)
    """User index 7."""

    @classmethod
    def from_ismrmrd_encoding_limits_type(cls, encoding_limits: encodingLimitsType):
        """Generate EncodingLimits from ismrmrd.encodingLimitsType."""
        values = {
            field.name: Limits.from_ismrmrd(getattr(encoding_limits, field.name))
            for field in dataclasses.fields(encoding_limits)
        }

        # adjust from ISMRMRD to MRPro naming convention
        values['k0'] = values.pop('kspace_encoding_step_0')
        values['k1'] = values.pop('kspace_encoding_step_1')
        values['k2'] = values.pop('kspace_encoding_step_2')

        return cls(**values)
