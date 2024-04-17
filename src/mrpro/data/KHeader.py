"""MR raw data / k-space data header dataclass."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses
import datetime
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema
import torch

from mrpro.data import enums
from mrpro.data.AcqInfo import AcqInfo
from mrpro.data.EncodingLimits import EncodingLimits
from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.data.TrajectoryDescription import TrajectoryDescription

if TYPE_CHECKING:
    # avoid circular imports by importing only when type checking
    from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator

UNKNOWN = 'unknown'


@dataclass(slots=True)
class KHeader(MoveDataMixin):
    """MR raw data header.

    All information that is not covered by the dataclass is stored in
    the misc dict Our code shall not rely on this information, and it is
    not guaranteed to be present Also, the information in the misc dict
    is not guaranteed to be correct or tested.
    """

    trajectory: KTrajectoryCalculator
    b0: float
    encoding_limits: EncodingLimits
    recon_matrix: SpatialDimension[int]
    recon_fov: SpatialDimension[float]
    encoding_matrix: SpatialDimension[int]
    encoding_fov: SpatialDimension[float]
    n_coils: int
    acq_info: AcqInfo
    datetime: datetime.datetime
    h1_freq: float
    te: torch.Tensor
    ti: torch.Tensor
    fa: torch.Tensor
    tr: torch.Tensor
    echo_spacing: torch.Tensor
    echo_train_length: int = 1
    seq_type: str = UNKNOWN
    model: str = UNKNOWN
    vendor: str = UNKNOWN
    protocol_name: str = UNKNOWN
    misc: dict = dataclasses.field(default_factory=dict)  # do not use {} here!
    calibration_mode: enums.CalibrationMode = enums.CalibrationMode.OTHER
    interleave_dim: enums.InterleavingDimension = enums.InterleavingDimension.OTHER
    traj_type: enums.TrajectoryType = enums.TrajectoryType.OTHER
    measurement_id: str = UNKNOWN
    patient_name: str = UNKNOWN
    trajectory_description: TrajectoryDescription = dataclasses.field(default_factory=TrajectoryDescription)

    @property
    def fa_degree(self) -> list[float]:
        """Flip angle in degree."""
        return [el / pi * 180 for el in self.fa]

    @classmethod
    def from_ismrmrd(
        cls,
        header: ismrmrdschema.ismrmrdHeader,
        acq_info: AcqInfo,
        defaults: dict | None = None,
        overwrite: dict | None = None,
        encoding_number: int = 0,
    ) -> KHeader:
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

        # Conversion functions for units
        def ms_to_s(ms: torch.Tensor) -> torch.Tensor:
            return ms / 1000

        def mm_to_m(m: float) -> float:
            return m / 1000

        if not 0 <= encoding_number < len(header.encoding):
            raise ValueError(f'encoding_number must be between 0 and {len(header.encoding)}')

        enc: ismrmrdschema.encodingType = header.encoding[encoding_number]

        # These are guaranteed to exist
        parameters = {'h1_freq': header.experimentalConditions.H1resonanceFrequency_Hz, 'acq_info': acq_info}

        if defaults is not None:
            parameters.update(defaults)

        if (
            header.acquisitionSystemInformation is not None
            and header.acquisitionSystemInformation.receiverChannels is not None
        ):
            parameters['n_coils'] = header.acquisitionSystemInformation.receiverChannels

        if header.sequenceParameters is not None:
            parameters['tr'] = ms_to_s(torch.as_tensor(header.sequenceParameters.TR))
            parameters['te'] = ms_to_s(torch.as_tensor(header.sequenceParameters.TE))
            parameters['ti'] = ms_to_s(torch.as_tensor(header.sequenceParameters.TI))
            parameters['fa'] = torch.deg2rad(torch.as_tensor(header.sequenceParameters.flipAngle_deg))
            parameters['echo_spacing'] = ms_to_s(torch.as_tensor(header.sequenceParameters.echo_spacing))

            if header.sequenceParameters.sequence_type is not None:
                parameters['seq_type'] = header.sequenceParameters.sequence_type

        if enc.reconSpace is not None:
            parameters['recon_fov'] = SpatialDimension[float].from_xyz(enc.reconSpace.fieldOfView_mm, mm_to_m)
            parameters['recon_matrix'] = SpatialDimension[int].from_xyz(enc.reconSpace.matrixSize)

        if enc.encodedSpace is not None:
            parameters['encoding_fov'] = SpatialDimension[float].from_xyz(enc.encodedSpace.fieldOfView_mm, mm_to_m)
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
            parameters['traj_type'] = enums.TrajectoryType(enc.trajectory.value)

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

            if header.acquisitionSystemInformation.systemFieldStrength_T is not None:
                parameters['b0'] = header.acquisitionSystemInformation.systemFieldStrength_T

        # estimate b0 from h1_freq if not given
        if 'b0' not in parameters:
            parameters['b0'] = parameters['h1_freq'] / 4258e4

        # Dump everything into misc
        parameters['misc'] = dataclasses.asdict(header)

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
