"""MR quantitative data header (QHeader) dataclass."""

from dataclasses import dataclass

from pydicom.dataset import Dataset
from pydicom.tag import Tag
from typing_extensions import Self

from mrpro.data.IHeader import IHeader
from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension


@dataclass(slots=True)
class QHeader(MoveDataMixin):
    """MR quantitative data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]
    """Field of view."""

    @classmethod
    def from_iheader(cls, iheader: IHeader) -> Self:
        """Create QHeader object from KHeader object.

        Parameters
        ----------
        iheader
            MR raw data header (IHeader) containing required meta data.
        """
        return cls(fov=iheader.fov)

    @classmethod
    def from_kheader(cls, kheader: KHeader) -> Self:
        """Create QHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """
        return cls(fov=kheader.recon_fov)

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

        fov_x_mm = float(get_items('Rows')[0]) * get_items('PixelSpacing')[0][0]
        fov_y_mm = float(get_items('Columns')[0]) * get_items('PixelSpacing')[0][1]
        fov_z_mm = float(get_items('SliceThickness')[0])
        fov = SpatialDimension(fov_x_mm, fov_y_mm, fov_z_mm) / 1000  # convert to m
        return cls(fov=fov)
