"""MR quantitative data header (QHeader) dataclass."""

from dataclasses import dataclass
from typing import Self

from mrpro.data.Header import Header
from mrpro.data.IHeader import IHeader
from mrpro.data.SpatialDimension import SpatialDimension


@dataclass(slots=True)
class QHeader(Header):
    """MR quantitative data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]
    """Field of view."""

    @classmethod
    def from_iheader(cls, iheader: IHeader) -> Self:
        """Create QHeader object from IHeader object.

        Parameters
        ----------
        iheader
            MR raw data header (IHeader) containing required meta data.
        """
        return cls(
            fov=iheader.fov,
            lamor_frequency_proton=iheader.lamor_frequency_proton,
            patient_table_position=iheader.patient_table_position,
            position=iheader.position,
            orientation=iheader.orientation,
            datetime=iheader.datetime,
            sequence_type=iheader.sequence_type,
            model=iheader.model,
            vendor=iheader.vendor,
            protocol_name=iheader.protocol_name,
            measurement_id=iheader.measurement_id,
            patient_name=iheader.patient_name,
            misc=iheader.misc,
        )
