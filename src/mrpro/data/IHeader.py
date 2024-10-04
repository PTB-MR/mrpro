"""MR image data header (IHeader) dataclass."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import torch
from pydicom.dataset import Dataset

from mrpro.data.Header import Header
from mrpro.data.KHeader import KHeader
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import ms_to_s

MISC_TAGS = {'TimeAfterStart': 0x00191016}
UNKNOWN = 'unknown'


@dataclass(slots=True)
class IHeader(Header):
    """MR image data header."""

    te: torch.Tensor | None
    """Echo time [s]."""

    ti: torch.Tensor | None
    """Inversion time [s]."""

    fa: torch.Tensor | None
    """Flip angle [rad]."""

    tr: torch.Tensor | None
    """Repetition time [s]."""

    echo_train_length: int | None = 1
    """Number of echoes in a multi-echo acquisition."""

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
            lamor_frequency_proton=kheader.lamor_frequency_proton,
            te=kheader.te,
            ti=kheader.ti,
            fa=kheader.fa,
            tr=kheader.tr,
            patient_table_position=kheader.acq_info.patient_table_position,
            position=kheader.acq_info.position,
            orientation=kheader.acq_info.orientation,
            datetime=kheader.datetime,
            echo_train_length=kheader.echo_train_length,
            sequence_type=kheader.sequence_type,
            model=kheader.model,
            vendor=kheader.vendor,
            protocol_name=kheader.protocol_name,
            measurement_id=kheader.measurement_id,
            patient_name=kheader.patient_name,
            misc={},
        )

    @classmethod
    def from_dicom_list(cls, dicom_datasets: Sequence[Dataset]) -> Self:
        """Read DICOM files and return IHeader object.

        Parameters
        ----------
        dicom_datasets
            list of dataset objects containing the DICOM file.
        """
        # Parameters which can vary between the entries of the dicom dataset
        fa = cls.make_unique_tensor(
            cls.get_items_as_list(dicom_datasets, 'FlipAngle', lambda item: np.deg2rad(float(item)))
        )
        ti = cls.make_unique_tensor(
            cls.get_items_as_list(dicom_datasets, 'InversionTime', lambda item: ms_to_s(float(item)))
        )
        tr = cls.make_unique_tensor(
            cls.get_items_as_list(dicom_datasets, 'RepetitionTime', lambda item: ms_to_s(float(item)))
        )

        # get echo time(s). Some scanners use 'EchoTime', some use 'EffectiveEchoTime'
        te_list = cls.get_items_as_list(dicom_datasets, 'EchoTime', lambda item: ms_to_s(float(item)))
        if all(val is None for val in te_list):  # check if all entries are None
            te_list = cls.get_items_as_list(dicom_datasets, 'EffectiveEchoTime', lambda item: ms_to_s(float(item)))
        te = cls.make_unique_tensor(te_list)
        # Get misc parameters
        misc = {}
        for name in MISC_TAGS:
            misc[name] = cls.make_unique_tensor(
                cls.get_items_as_list(dicom_datasets, MISC_TAGS[name], lambda item: float(item))
            )
        return cls(
            te=te,
            ti=ti,
            fa=fa,
            tr=tr,
            misc=misc,
            **cls.attributes_from_dicom_list(dicom_datasets),
        )

    def __repr__(self):
        """Representation method for IHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = f'FOV [m]: {self.fov!s}\n' f'TE [s]: {te}\nTI [s]: {ti}\nFlip angle [rad]: {fa}.'
        return out
