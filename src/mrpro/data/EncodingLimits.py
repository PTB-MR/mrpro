"""Encoding limits dataclass."""

import dataclasses
from dataclasses import field

from ismrmrd.xsd.ismrmrdschema.ismrmrd import encodingLimitsType, encodingType, ismrmrdHeader, limitType
from typing_extensions import Self


@dataclasses.dataclass(slots=True)
class Limits:
    """Limits dataclass with min, max, and center attributes."""

    min: int = 0
    """Lower boundary."""

    max: int = 0
    """Upper boundary."""

    center: int = 0
    """Center."""

    @classmethod
    def from_ismrmrd(cls, limit_type: limitType | None) -> Self:
        """Create Limits from ismrmrd.limitType."""
        if limit_type is None:
            return cls()
        return cls(*dataclasses.astuple(limit_type))

    @property
    def length(self) -> int:
        """Length of the limits."""
        return self.max - self.min + 1


@dataclasses.dataclass(slots=True)
class EncodingLimits:
    """Encoding limits dataclass with limits for each attribute [INA2016]_.

    References
    ----------
    .. [INA2016] Inati S, Hansen M (2016) ISMRM Raw data format: A proposed standard for MRI raw datasets. MRM 77(1)
        https://doi.org/10.1002/mrm.26089

    """

    k0: Limits = field(default_factory=Limits)
    """First k-space encoding."""

    k1: Limits = field(default_factory=Limits)
    """Second k-space encoding."""

    k2: Limits = field(default_factory=Limits)
    """Third k-space encoding."""

    average: Limits = field(default_factory=Limits)
    """Signal average."""

    slice: Limits = field(default_factory=Limits)
    """Slice number (multi-slice 2D)."""

    contrast: Limits = field(default_factory=Limits)
    """Echo number in multi-echo."""

    phase: Limits = field(default_factory=Limits)
    """Cardiac phase."""

    repetition: Limits = field(default_factory=Limits)
    """Repeated/dynamic acquisitions."""

    set: Limits = field(default_factory=Limits)
    """Sets of different preparation."""

    segment: Limits = field(default_factory=Limits)
    """Segments of segmented acquisition."""

    user0: Limits = field(default_factory=Limits)
    """User index 0."""

    user1: Limits = field(default_factory=Limits)
    """User index 1."""

    user2: Limits = field(default_factory=Limits)
    """User index 2."""

    user3: Limits = field(default_factory=Limits)
    """User index 3."""

    user4: Limits = field(default_factory=Limits)
    """User index 4."""

    user5: Limits = field(default_factory=Limits)
    """User index 5."""

    user6: Limits = field(default_factory=Limits)
    """User index 6."""

    user7: Limits = field(default_factory=Limits)
    """User index 7."""

    @classmethod
    def from_ismrmrd_header(
        cls,
        header: ismrmrdHeader,
        encoding_number: int = 0,
    ) -> Self:
        """
        Extract EncodingLimits from ismrmrd.ismrmrdHeader.

        Parameters
        ----------
        header
            ISMRMRD header
        encoding_number
            Encoding number. An ValueError is raised if the encoding number is out of range for the header.

        Returns
        -------
        Extracted EncodingLimits if header.encoding.encodingLimits is not None,
        otherwise an empty EncodingLimits.
        """
        if not 0 <= encoding_number < len(header.encoding):
            raise ValueError(f'encoding_number must be between 0 and {len(header.encoding)}')
        enc: encodingType = header.encoding[encoding_number]
        if enc.encodingLimits is None:
            return cls()
        return cls.from_ismrmrd_encoding_limits_type(enc.encodingLimits)

    @classmethod
    def from_ismrmrd_encoding_limits_type(cls, encoding_limits: encodingLimitsType) -> Self:
        """Generate EncodingLimits from ismrmrd.encodingLimitsType."""
        values = {
            field.name: Limits.from_ismrmrd(getattr(encoding_limits, field.name))
            for field in dataclasses.fields(encoding_limits)
        }

        # adjust from ISMRMRD to MRPro naming convention
        values['k0'] = values.pop('kspace_encoding_step_0')
        values['k1'] = values.pop('kspace_encoding_step_1')
        values['k2'] = values.pop('kspace_encoding_step_2')
        values['user0'] = values.pop('user_0')
        values['user1'] = values.pop('user_1')
        values['user2'] = values.pop('user_2')
        values['user3'] = values.pop('user_3')
        values['user4'] = values.pop('user_4')
        values['user5'] = values.pop('user_5')
        values['user6'] = values.pop('user_6')
        values['user7'] = values.pop('user_7')

        return cls(**values)
