"""MR raw data / k-space data header dataclass."""

from __future__ import annotations

import dataclasses
import datetime
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema
import torch
from typing_extensions import Self

from mrpro.data import enums
from mrpro.data.AcqInfo import AcqInfo
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.summarize_tensorvalues import summarize_tensorvalues
from mrpro.utils.unit_conversion import (
    deg_to_rad,
    lamor_frequency_to_magnetic_field,
    m_to_mm,
    mm_to_m,
    ms_to_s,
    rad_to_deg,
    s_to_ms,
)

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

    recon_matrix: SpatialDimension[int]
    """Dimensions of the reconstruction matrix."""

    encoding_matrix: SpatialDimension[int]
    """Dimensions of the encoded k-space matrix."""

    recon_fov: SpatialDimension[float]
    """Field-of-view of the reconstructed image [m]."""

    encoding_fov: SpatialDimension[float]
    """Field of view of the image encoded by the k-space trajectory [m]."""

    acq_info: AcqInfo = field(default_factory=AcqInfo)
    """Information of the acquisitions (i.e. readout lines)."""

    trajectory: KTrajectoryCalculator | None = None
    """Function to calculate the k-space trajectory."""

    lamor_frequency_proton: float | None = None
    """Lamor frequency of hydrogen nuclei [Hz]."""

    datetime: datetime.datetime | None = None
    """Date and time of acquisition."""

    te: list[float] | torch.Tensor = field(default_factory=list)
    """Echo time [s]."""

    ti: list[float] | torch.Tensor = field(default_factory=list)
    """Inversion time [s]."""

    fa: list[float] | torch.Tensor = field(default_factory=list)
    """Flip angle [rad]."""

    tr: list[float] | torch.Tensor = field(default_factory=list)
    """Repetition time [s]."""

    echo_spacing: list[float] | torch.Tensor = field(default_factory=list)
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

    trajectory_type: enums.TrajectoryType = enums.TrajectoryType.OTHER
    """Type of trajectory."""

    measurement_id: str = UNKNOWN
    """Measurement ID."""

    patient_name: str = UNKNOWN
    """Name of the patient."""

    _misc: dict = field(default_factory=dict)
    """Dictionary with miscellaneous parameters. These parameters are for information purposes only. Reconstruction
    algorithms should not rely on them."""

    @property
    def fa_degree(self) -> torch.Tensor | list[float]:
        """Flip angle in degree."""
        if isinstance(self.fa, torch.Tensor):
            return torch.rad2deg(self.fa)
        elif not len(self.fa):
            warnings.warn('Flip angle is not defined.', stacklevel=1)
            return []
        else:
            return torch.rad2deg(torch.as_tensor(self.fa)).tolist()

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
                parameters['tr'] = ms_to_s(header.sequenceParameters.TR)
            if header.sequenceParameters.TE:
                parameters['te'] = ms_to_s(header.sequenceParameters.TE)
            if header.sequenceParameters.TI:
                parameters['ti'] = ms_to_s(header.sequenceParameters.TI)
            if header.sequenceParameters.flipAngle_deg:
                parameters['fa'] = deg_to_rad(header.sequenceParameters.flipAngle_deg)
            if header.sequenceParameters.echo_spacing:
                parameters['echo_spacing'] = ms_to_s(header.sequenceParameters.echo_spacing)

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

        if enc.echoTrainLength is not None:
            parameters['echo_train_length'] = enc.echoTrainLength

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

    def to_ismrmrd(self) -> ismrmrdschema.ismrmrdHeader:
        """Create ISMRMRD header from KHeader.

        All the parameters from the ISMRMRD header of the raw file used to create the kheader are saved in the
        dictionary kheader['misc']. We first fill the information from this dictionary into the ISMRMRD header and then
        we update the values based on the attributes of the KHeader.

        Returns
        -------
            ISMRMRD header
        """
        header = ismrmrdschema.ismrmrdHeader()

        # Experimental conditions
        exp = ismrmrdschema.experimentalConditionsType()
        exp.H1resonanceFrequency_Hz = self.lamor_frequency_proton
        header.experimentalConditions = exp

        # Subject information
        subj = ismrmrdschema.subjectInformationType()
        subj.patientName = self.patient_name
        header.subjectInformation = subj

        # Measurement information
        meas = ismrmrdschema.measurementInformationType()
        meas.protocolName = self.protocol_name
        meas.measurementID = self.measurement_id
        if self.datetime is not None:
            meas.seriesTime = str(self.datetime.time())
            meas.seriesDate = str(self.datetime.date())
        header.measurementInformation = meas

        # Acquisition system information
        sys = ismrmrdschema.acquisitionSystemInformationType()
        sys.systemModel = self.model
        sys.systemVendor = self.vendor
        if self.lamor_frequency_proton:
            sys.systemFieldStrength_T = lamor_frequency_to_magnetic_field(self.lamor_frequency_proton)
        header.acquisitionSystemInformation = sys

        # Sequence information
        seq = ismrmrdschema.sequenceParametersType()
        if self.tr is not None:
            seq.TR = s_to_ms(self.tr)
        if self.te is not None:
            seq.TE = s_to_ms(self.te)
        if self.ti is not None:
            seq.TI = s_to_ms(self.ti)
        if self.fa is not None:
            seq.flipAngle_deg = rad_to_deg(self.fa)
        if self.echo_spacing is not None:
            seq.echo_spacing = s_to_ms(self.echo_spacing)
        seq.sequence_type = self.sequence_type
        header.sequenceParameters = seq

        # Encoding
        encoding = ismrmrdschema.encodingType()

        # Encoded space
        encoding_space = ismrmrdschema.encodingSpaceType()
        encoding_fov = ismrmrdschema.fieldOfViewMm(
            m_to_mm(self.encoding_fov.x), m_to_mm(self.encoding_fov.y), m_to_mm(self.encoding_fov.z)
        )
        encoding_matrix = ismrmrdschema.matrixSizeType(
            self.encoding_matrix.x, self.encoding_matrix.y, self.encoding_matrix.z
        )
        encoding_space.matrixSize = encoding_matrix
        encoding_space.fieldOfView_mm = encoding_fov

        # Recon space
        recon_space = ismrmrdschema.encodingSpaceType()
        recon_fov = ismrmrdschema.fieldOfViewMm(
            m_to_mm(self.recon_fov.x), m_to_mm(self.recon_fov.y), m_to_mm(self.recon_fov.z)
        )
        recon_matrix = ismrmrdschema.matrixSizeType(self.recon_matrix.x, self.recon_matrix.y, self.recon_matrix.z)
        recon_space.matrixSize = recon_matrix
        recon_space.fieldOfView_mm = recon_fov

        # Set encoded and recon spaces
        encoding.encodedSpace = encoding_space
        encoding.reconSpace = recon_space
        header.encoding.append(encoding)

        return header

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
