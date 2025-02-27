"""MR image data header (IHeader) dataclass."""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np
import torch
from pydicom import multival, valuerep
from pydicom.dataset import Dataset
from pydicom.tag import Tag, TagType
from typing_extensions import Self

from mrpro.data.KHeader import KHeader
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import deg_to_rad, mm_to_m, ms_to_s

MISC_TAGS = {'TimeAfterStart': 0x00191016}


@dataclass(slots=True)
class IHeader(MoveDataMixin):
    """MR image data header."""

    # ToDo: decide which attributes to store in the header
    fov: SpatialDimension[float]
    """Field of view [m]."""

    te: torch.Tensor | None
    """Echo time [s]."""

    ti: torch.Tensor | None
    """Inversion time [s]."""

    fa: torch.Tensor | None
    """Flip angle [rad]."""

    tr: torch.Tensor | None
    """Repetition time [s]."""

    misc: dict = dataclasses.field(default_factory=dict)
    """Dictionary with miscellaneous parameters."""

    @classmethod
    def from_kheader(cls, kheader: KHeader) -> Self:
        """Create IHeader object from KHeader object.

        Parameters
        ----------
        kheader
            MR raw data header (KHeader) containing required meta data.
        """
        return cls(
            fov=kheader.recon_fov,
            te=torch.as_tensor(kheader.te),
            ti=torch.as_tensor(kheader.ti),
            fa=torch.as_tensor(kheader.fa),
            tr=torch.as_tensor(kheader.tr),
        )

    @classmethod
    def from_dicom_list(cls, dicom_datasets: Sequence[Dataset]) -> Self:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """

        def convert_pydicom_type(item):  # noqa: ANN001
            """Convert item to type given by pydicom."""
            if isinstance(item, valuerep.DSfloat | valuerep.DSdecimal):
                return float(item)
            elif isinstance(item, valuerep.IS | valuerep.ISfloat):
                return int(item)
            elif isinstance(item, multival.MultiValue):
                return [convert_pydicom_type(it) for it in item]
            else:
                return item

        def get_items_from_dataset(dataset: Dataset, name: TagType) -> Sequence:
            """Get item with a given name or Tag from pydicom dataset."""
            # iterall is recursive, so it will find all items with the given name
            found_item = [convert_pydicom_type(item.value) for item in dataset.iterall() if item.tag == Tag(name)]
            if len(found_item) == 0:
                return (None,)
            return found_item

        def get_items(datasets: Sequence[Dataset], name: TagType):
            """Get list of items for all dataset objects in the list."""
            return list(chain.from_iterable([get_items_from_dataset(ds, name) for ds in datasets]))

        def make_unique_tensor(values: Sequence[float]) -> torch.Tensor | None:
            """If all the values are the same only return one."""
            if any(val is None for val in values):
                return None
            elif len(np.unique(values)) == 1:
                return torch.as_tensor([values[0]])
            else:
                return torch.as_tensor(values)

        # Ensure datasets are consistently single-frame or multi-frame 2D/3D
        number_of_frames = get_items(dicom_datasets, 'NumberOfFrames')
        if len(set(number_of_frames)) > 1:
            raise ValueError('Only DICOM files with the same number of frames can be stacked.')
        mr_acquisition_type = get_items(dicom_datasets, 'MRAcquisitionType')
        if len(set(mr_acquisition_type)) > 1:
            raise ValueError('Only DICOM files with the same MRAcquisitionType can be stacked.')

        # Check if the data is multi-frame 3D
        datasets_are_3d = False
        if number_of_frames[0] is not None and number_of_frames[0] > 1 and mr_acquisition_type[0] == '3D':
            datasets_are_3d = True

        # Calculate FOV
        image_rows = get_items(dicom_datasets, 'Rows')
        if len(set(image_rows)) > 1 or image_rows[0] is None:
            raise ValueError('Number of Rows needs to be defined and the same for all.')
        image_columns = get_items(dicom_datasets, 'Columns')
        if len(set(image_columns)) > 1 or image_columns[0] is None:
            raise ValueError('Number of Columns needs to be defined and the same for all.')
        pixel_spacing = get_items(dicom_datasets, 'PixelSpacing')
        if pixel_spacing[0] is None or np.unique(np.asarray(pixel_spacing), axis=0).shape[0] > 1:
            raise ValueError('Pixel spacing needs to be defined and the same for all.')
        slice_thickness = get_items(dicom_datasets, 'SliceThickness')
        if len(set(slice_thickness)) > 1 or slice_thickness[0] is None:
            raise ValueError('Slice thickness needs to be defined and the same for all.')

        fov = SpatialDimension(
            z=slice_thickness[0] * number_of_frames[0] if datasets_are_3d else slice_thickness[0],
            y=image_columns[0] * pixel_spacing[0][1],
            x=image_rows[0] * pixel_spacing[0][0],
        ).apply_(mm_to_m)

        # Parameters which are optional
        # For 3D datasets these parameters are the same for each 3D volume so we only keep one value per volume
        n_volumes = number_of_frames[0] if datasets_are_3d else 1
        fa_deg = make_unique_tensor(get_items(dicom_datasets, 'FlipAngle')[::n_volumes])
        ti_ms = make_unique_tensor(get_items(dicom_datasets, 'InversionTime')[::n_volumes])
        tr_ms = make_unique_tensor(get_items(dicom_datasets, 'RepetitionTime')[::n_volumes])

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_ms = get_items(dicom_datasets, 'EchoTime')
        if all(val is None for val in te_ms):  # check if all entries are None
            te_ms = get_items(dicom_datasets, 'EffectiveEchoTime')
        te_ms = make_unique_tensor(te_ms[::n_volumes])

        # Get misc parameters
        misc = {}
        for name in MISC_TAGS:
            misc[name] = make_unique_tensor(get_items(dicom_datasets, MISC_TAGS[name]))

        return cls(
            fov=fov,
            fa=None if fa_deg is None else deg_to_rad(fa_deg),
            ti=None if ti_ms is None else ms_to_s(ti_ms),
            tr=None if tr_ms is None else ms_to_s(tr_ms),
            te=None if te_ms is None else ms_to_s(te_ms),
            misc=misc,
        )

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = f'FOV [m]: {self.fov!s}\nTE [s]: {te}\nTI [s]: {ti}\nFlip angle [rad]: {fa}.'
        return out
