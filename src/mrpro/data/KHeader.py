"""MR raw data / k-space data header dataclass."""

from __future__ import annotations

import dataclasses
import datetime
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema
import torch
from typing_extensions import Self

from mrpro.data import enums
from mrpro.data.AcqInfo import AcqInfo
from mrpro.data.EncodingLimits import EncodingLimits
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import mm_to_m, ms_to_s

if TYPE_CHECKING:
    # avoid circular imports by importing only when type checking
    from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator

UNKNOWN = 'unknown'


@dataclass(slots=True)
class KHeader(MoveDataMixin):
    """MR raw data header.

    All information that is not covered by the dataclass is stored in
    the misc dict. Our code shall not rely on this information, and it is
    not guaranteed to be present. Also, the information in the misc dict
    is not guaranteed to be correct or tested.
    """

    trajectory: KTrajectoryCalculator
    """Function to calculate the k-space trajectory."""

    encoding_limits: EncodingLimits
    """K-space encoding limits."""

    recon_matrix: SpatialDimension[int]
    """Dimensions of the reconstruction matrix."""

    recon_fov: SpatialDimension[float]
    """Field-of-view of the reconstructed image [m]."""

    encoding_matrix: SpatialDimension[int]
    """Dimensions of the encoded k-space matrix."""

    encoding_fov: SpatialDimension[float]
    """Field of view of the image encoded by the k-space trajectory [m]."""

    acq_info: AcqInfo
    """Information of the acquisitions (i.e. readout lines)."""

    lamor_frequency_proton: float
    """Lamor frequency of hydrogen nuclei [Hz]."""

    datetime: datetime.datetime | None = None
    """Date and time of acquisition."""

    te: torch.Tensor | None = None
    """Echo time [s]."""

    ti: torch.Tensor | None = None
    """Inversion time [s]."""

    fa: torch.Tensor | None = None
    """Flip angle [rad]."""

    tr: torch.Tensor | None = None
    """Repetition time [s]."""

    echo_spacing: torch.Tensor | None = None
    """Echo spacing [s]."""

    echo_train_length: int = 1
    """Number of echoes in a multi-echo acquisition."""

    sequence_type: str = UNKNOWN
    """Type of sequence."""

    model: str = UNKNOWN
    """Scanner model."""

    vendor: str = UNKNOWN
    """Scanner vendor."""

    protocol_name: str = UNKNOWN
    """Name of the acquisition protocol."""

    calibration_mode: enums.CalibrationMode = enums.CalibrationMode.OTHER
    """Mode of how calibration data is acquired. """

    interleave_dim: enums.InterleavingDimension = enums.InterleavingDimension.OTHER
    """Interleaving dimension."""

    trajectory_type: enums.TrajectoryType = enums.TrajectoryType.OTHER
    """Type of trajectory."""

    measurement_id: str = UNKNOWN
    """Measurement ID."""

    patient_name: str = UNKNOWN
    """Name of the patient."""

    _misc: dict = dataclasses.field(default_factory=dict)  # do not use {} here!
    """Dictionary with miscellaneous parameters. These parameters are for information purposes only. Reconstruction
    algorithms should not rely on them."""

    @property
    def fa_degree(self) -> torch.Tensor | None:
        """Flip angle in degree."""
        if self.fa is None:
            warnings.warn('Flip angle is not defined.', stacklevel=1)
            return None
        else:
            return torch.rad2deg(self.fa)

    @classmethod
    def from_ismrmrd(
        cls,
        header: ismrmrdschema.ismrmrdHeader,
        acq_info: AcqInfo,
        defaults: dict | None = None,
        overwrite: dict | None = None,
        encoding_number: int = 0,
    ) -> Self:
        """Create an Header from ISMRMRD Data.

        Parameters
        ----------
        header
            ISMRMRD header
        acq_info
            acquisition information
        defaults
            dictionary of values to be used if information is missing in header
        overwrite
            dictionary of values to be used independent of header
        encoding_number
            as ismrmrdHeader can contain multiple encodings, selects which to consider
        """
        if not 0 <= encoding_number < len(header.encoding):
            raise ValueError(f'encoding_number must be between 0 and {len(header.encoding)}')

        enc: ismrmrdschema.encodingType = header.encoding[encoding_number]

        # These are guaranteed to exist
        parameters = {
            'lamor_frequency_proton': header.experimentalConditions.H1resonanceFrequency_Hz,
            'acq_info': acq_info,
        }

        if defaults is not None:
            parameters.update(defaults)

        if header.sequenceParameters is not None:
            if header.sequenceParameters.TR:
                parameters['tr'] = ms_to_s(torch.as_tensor(header.sequenceParameters.TR))
            if header.sequenceParameters.TE:
                parameters['te'] = ms_to_s(torch.as_tensor(header.sequenceParameters.TE))
            if header.sequenceParameters.TI:
                parameters['ti'] = ms_to_s(torch.as_tensor(header.sequenceParameters.TI))
            if header.sequenceParameters.flipAngle_deg:
                parameters['fa'] = torch.deg2rad(torch.as_tensor(header.sequenceParameters.flipAngle_deg))
            if header.sequenceParameters.echo_spacing:
                parameters['echo_spacing'] = ms_to_s(torch.as_tensor(header.sequenceParameters.echo_spacing))

            if header.sequenceParameters.sequence_type is not None:
                parameters['sequence_type'] = header.sequenceParameters.sequence_type

        if enc.reconSpace is not None:
            parameters['recon_fov'] = SpatialDimension[float].from_xyz(enc.reconSpace.fieldOfView_mm).apply_(mm_to_m)
            parameters['recon_matrix'] = SpatialDimension[int].from_xyz(enc.reconSpace.matrixSize)

        if enc.encodedSpace is not None:
            parameters['encoding_fov'] = (
                SpatialDimension[float].from_xyz(enc.encodedSpace.fieldOfView_mm).apply_(mm_to_m)
            )
            parameters['encoding_matrix'] = SpatialDimension[int].from_xyz(enc.encodedSpace.matrixSize)

        if enc.encodingLimits is not None:
            parameters['encoding_limits'] = EncodingLimits.from_ismrmrd_encoding_limits_type(enc.encodingLimits)

        if enc.echoTrainLength is not None:
            parameters['echo_train_length'] = enc.echoTrainLength

        if enc.parallelImaging is not None:
            if enc.parallelImaging.calibrationMode is not None:
                parameters['calibration_mode'] = enums.CalibrationMode(enc.parallelImaging.calibrationMode.value)

            if enc.parallelImaging.interleavingDimension is not None:
                parameters['interleave_dim'] = enums.InterleavingDimension(
                    enc.parallelImaging.interleavingDimension.value,
                )

        if enc.trajectory is not None:
            parameters['trajectory_type'] = enums.TrajectoryType(enc.trajectory.value)

        # Either use the series or study time if available
        if header.measurementInformation is not None and header.measurementInformation.seriesTime is not None:
            time = header.measurementInformation.seriesTime.to_time()
        elif header.studyInformation is not None and header.studyInformation.studyTime is not None:
            time = header.studyInformation.studyTime.to_time()
        else:  # if no time is given, set to 00:00:00
            time = datetime.time()
        if header.measurementInformation is not None and header.measurementInformation.seriesDate is not None:
            date = header.measurementInformation.seriesDate.to_date()
            parameters['datetime'] = datetime.datetime.combine(date, time)
        elif header.studyInformation is not None and header.studyInformation.studyDate is not None:
            date = header.studyInformation.studyDate.to_date()
            parameters['datetime'] = datetime.datetime.combine(date, time)

        if header.subjectInformation is not None and header.subjectInformation.patientName is not None:
            parameters['patient_name'] = header.subjectInformation.patientName

        if header.measurementInformation is not None:
            if header.measurementInformation.measurementID is not None:
                parameters['measurement_id'] = header.measurementInformation.measurementID

            if header.measurementInformation.protocolName is not None:
                parameters['protocol_name'] = header.measurementInformation.protocolName

        if header.acquisitionSystemInformation is not None:
            if header.acquisitionSystemInformation.systemVendor is not None:
                parameters['vendor'] = header.acquisitionSystemInformation.systemVendor

            if header.acquisitionSystemInformation.systemModel is not None:
                parameters['model'] = header.acquisitionSystemInformation.systemModel

        # Dump everything into misc
        parameters['_misc'] = dataclasses.asdict(header)

        if overwrite is not None:
            parameters.update(overwrite)

        try:
            instance = cls(**parameters)
        except TypeError:
            missing = [
                f.name
                for f in dataclasses.fields(cls)
                if f.name not in parameters
                and (f.default == dataclasses.MISSING and f.default_factory == dataclasses.MISSING)
            ]
            raise ValueError(
                f'Could not create Header. Missing parameters: {missing}\n'
                'Consider setting them via the defaults dictionary',
            ) from None
        return instance

    def __repr__(self):
        """Representation method for KHeader class."""
        te = summarize_tensorvalues(self.te)
        ti = summarize_tensorvalues(self.ti)
        fa = summarize_tensorvalues(self.fa)
        out = (
            f'FOV [m]: {self.encoding_fov!s}\n'
            f'TE [s]: {te}\n'
            f'TI [s]: {ti}\n'
            f'Flip angle [rad]: {fa}\n'
            f'Encoding matrix: {self.encoding_matrix!s} \n'
            f'Recon matrix: {self.recon_matrix!s} \n'
        )
        return out
