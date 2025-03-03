"""MR quantitative data header (QHeader) dataclass."""

from dataclasses import dataclass, field

import torch
from pydicom.dataset import Dataset
from pydicom.tag import Tag
from typing_extensions import Self

from mrpro.data.IHeader import IHeader, ImageIdx
from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.Rotation import Rotation
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.unit_conversion import mm_to_m


@dataclass(slots=True)
class QHeader(MoveDataMixin):
    """MR quantitative data header."""

    resolution: SpatialDimension[float]
    """Resolution [m/px] of the data."""

    position: SpatialDimension[torch.Tensor] = field(
        default_factory=lambda: SpatialDimension(
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
            torch.zeros(1, 1, 1, 1, 1),
        )
    )
    """Position of the data"""

    orientation: Rotation = field(default_factory=lambda: Rotation.identity((1, 1, 1, 1, 1)))
    """Orientation of the data"""

    idx: ImageIdx = field(default_factory=ImageIdx)

    @classmethod
    def from_iheader(cls, header: IHeader) -> Self:
        """Create QHeader object from IHeader object.

        Parameters
        ----------
        header
            MR raw data header (IHeader) containing required meta data.
        """
        return cls(resolution=header.resolution, position=header.position, orientation=header.orientation)

    @classmethod
    def from_kheader(cls, header: KHeader) -> Self:
        """Create QHeader object from KHeader object.

        Parameters
        ----------
        header
            MR raw data header (KHeader) containing required meta data.
        """
        resolution = header.recon_fov / header.recon_matrix
        return cls(resolution=resolution, position=header.acq_info.position, orientation=header.acq_info.orientation)

    @classmethod
    def from_dicom(cls, dicom_dataset: Dataset) -> Self:
        """Read DICOM file containing qMRI data and return QHeader object.

        Parameters
        ----------
        dicom_dataset
            dataset object containing the DICOM file.
        """

        def get_items(name: str):  # Todo: move to utils and reuse logic in IHeader
            """Get all items with a given name from a pydicom dataset."""
            # iterall is recursive, so it will find all items with the given name
            return [item.value for item in dicom_dataset.iterall() if item.tag == Tag(name)]

        resolution = SpatialDimension(
            get_items('SliceThickness')[0], get_items('PixelSpacing')[0][1], get_items('PixelSpacing')[0][0]
        ).apply_(mm_to_m)
        return cls(resolution=resolution)
