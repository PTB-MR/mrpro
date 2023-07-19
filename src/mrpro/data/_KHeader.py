"""Data classes for MR raw data header."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema

from mrpro.data import AcqInfo
from mrpro.data.raw import ELimits
from mrpro.data.raw import return_acc_factor_tensor
from mrpro.data.raw import return_coil_label_dict
from mrpro.data.raw import return_par_enc_limits_tensor
from mrpro.data.raw import return_par_matrix_tensor


class KHeader:
    __slots__ = (
        'protocol_name',
        'pat_pos',
        'meas_id',
        'institution',
        'receiver_noise_bwdth',
        'b0',
        'model',
        'vendor',
        'elimits',
        'acc_factor',
        'rec_matrix',
        'rec_fov',
        'enc_matrix',
        'enc_fov',
        'etl',
        'num_coils',
        'acq_info',
        'calib_mode',
        'interleave_dim',
        'multiband',
        'traj_type',
        'traj_description',
        'device_id',
        'device_sn',
        'coil_label',
        'station_name',
        'h1freq',
        'ref_frame_uid',
        'series_num',
        'meas_depend',
        'ref_im_seq',
        'rel_table_pos',
        'seq_name',
        'series_date',
        'series_description',
        'series_uid',
        'series_time',
        'te',
        'ti',
        'tr_siemens',
        'fa',
        'tr',
        'seq_type',
        'diffusion',
        'diffusion_dim',
        'diffusion_scheme',
        'accession_number',
        'body_part',
        'ref_physician',
        'study_date',
        'study_description',
        'study_id',
        'study_uid',
        'study_time',
    )

    def __init__(self) -> None:
        pass

    def from_ismrmrd_header(
        self, header: ismrmrdschema.ismrmrdHeader, num_acq: int
    ) -> None:
        # Encoding
        assert (
            len(header.encoding) == 1
        ), 'Multiple encodings are not supported.'
        enc = header.encoding[0]
        self.etl = enc.echoTrainLength
        self.enc_fov = return_par_matrix_tensor(
            enc.encodedSpace.fieldOfView_mm
        )
        self.enc_matrix = return_par_matrix_tensor(enc.encodedSpace.matrixSize)
        self.rec_fov = return_par_matrix_tensor(enc.reconSpace.fieldOfView_mm)
        self.rec_matrix = return_par_matrix_tensor(enc.reconSpace.matrixSize)
        self.acc_factor = return_acc_factor_tensor(
            enc.parallelImaging.accelerationFactor
        )
        self.calib_mode = enc.parallelImaging.calibrationMode.value
        self.interleave_dim = enc.parallelImaging.interleavingDimension.value
        self.multiband = enc.parallelImaging.multiband
        self.traj_type = enc.trajectory.value
        self.traj_description = enc.trajectoryDescription

        self.elimits = ELimits()
        for climit in self.elimits.__slots__:
            setattr(
                self.elimits,
                climit,
                return_par_enc_limits_tensor(
                    getattr(enc.encodingLimits, climit)
                ),
            )

        # AcquisitionSystemInformation
        self.vendor = header.acquisitionSystemInformation.systemVendor
        self.model = header.acquisitionSystemInformation.systemModel
        self.station_name = header.acquisitionSystemInformation.stationName
        self.device_id = header.acquisitionSystemInformation.deviceID
        self.device_sn = header.acquisitionSystemInformation.deviceSerialNumber
        self.coil_label = return_coil_label_dict(
            header.acquisitionSystemInformation.coilLabel
        )
        self.b0 = header.acquisitionSystemInformation.systemFieldStrength_T
        self.receiver_noise_bwdth = (
            header.acquisitionSystemInformation.relativeReceiverNoiseBandwidth
        )
        self.institution = header.acquisitionSystemInformation.institutionName
        self.num_coils = header.acquisitionSystemInformation.receiverChannels

        # ExperimentalConditions
        self.h1freq = header.experimentalConditions.H1resonanceFrequency_Hz

        # MeasurementInformation
        self.ref_frame_uid = header.measurementInformation.frameOfReferenceUID
        self.series_num = header.measurementInformation.initialSeriesNumber
        self.meas_depend = header.measurementInformation.measurementDependency
        self.meas_id = header.measurementInformation.measurementID
        self.pat_pos = header.measurementInformation.patientPosition.value
        self.protocol_name = header.measurementInformation.protocolName
        self.ref_im_seq = header.measurementInformation.referencedImageSequence
        self.rel_table_pos = (
            header.measurementInformation.relativeTablePosition
        )
        self.seq_name = header.measurementInformation.sequenceName
        self.series_date = header.measurementInformation.seriesDate
        self.series_description = (
            header.measurementInformation.seriesDescription
        )
        self.series_uid = header.measurementInformation.seriesInstanceUIDRoot
        self.series_time = header.measurementInformation.seriesTime

        # SequenceParameters
        self.te = header.sequenceParameters.TE
        self.ti = header.sequenceParameters.TI
        self.tr_siemens = header.sequenceParameters.TR
        self.fa = header.sequenceParameters.flipAngle_deg
        self.tr = header.sequenceParameters.echo_spacing
        self.seq_type = header.sequenceParameters.sequence_type
        self.diffusion = header.sequenceParameters.diffusion
        self.diffusion_dim = header.sequenceParameters.diffusionDimension
        self.diffusion_scheme = header.sequenceParameters.diffusionScheme

        # StudyInformation
        self.accession_number = header.studyInformation.accessionNumber
        self.body_part = header.studyInformation.bodyPartExamined
        self.ref_physician = header.studyInformation.referringPhysicianName
        self.study_date = header.studyInformation.studyDate
        self.study_description = header.studyInformation.studyDescription
        self.study_id = header.studyInformation.studyID
        self.study_uid = header.studyInformation.studyInstanceUID
        self.study_time = header.studyInformation.studyTime

        # Acquisition info will be filled during reading in each readout
        self.acq_info: AcqInfo = AcqInfo(num_acq)
