from __future__ import annotations

from dataclasses import dataclass

from ismrmrd.xsd.ismrmrdschema.ismrmrd import accelerationFactorType


@dataclass(slots=True)
class AccelerationFactor:
    """Acceleration Factor."""

    kspace_encoding_step_1: float
    kspace_encoding_step_2: float

    @property
    def overall(self) -> float:
        return self.kspace_encoding_step_1 * self.kspace_encoding_step_2

    @classmethod
    def from_ismrmrd(cls, data: accelerationFactorType) -> AccelerationFactor:
        """Create a AccelerationFactor from ismrmrd accelerationFactorType."""
        return cls(data.kspace_encoding_step_1, data.kspace_encoding_step_1)
