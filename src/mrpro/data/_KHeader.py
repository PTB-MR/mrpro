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

from dataclasses import dataclass

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema
import torch

from mrpro.data import AcqInfo
from mrpro.data import EncodingLimits
from mrpro.data.raw import return_acc_factor_tensor
from mrpro.data.raw import return_coil_label_dict
from mrpro.data.raw import return_par_matrix_tensor


@dataclass(slots=True)
class KHeader:
    protocol_name: str
    pat_pos: str
    meas_id: int
    institution: str
    receiver_noise_bwdth: float
    b0: float
    model: str
    vendor: str
    encoding_limits: EncodingLimits
    acc_factor: torch.Tensor | None
    recon_matrix: torch.Tensor | None
    recon_fov: torch.Tensor | None
    encoding_matrix: torch.Tensor | None
    encoding_fov: torch.Tensor | None
    echo_train_length: int
    num_coils: int
    acq_info: AcqInfo
    calib_mode: str
    interleave_dim: str
    multiband: bool
    traj_type: str
    traj_description: str
    device_id: str
    device_sn: str
    coil_label: dict[str, str]
    station_name: str
    h1freq: float
    ref_frame_uid: str
    series_num: int
    meas_depend: str
    ref_im_seq: str
    rel_table_pos: float
    seq_name: str
    series_date: str
    series_description: str
    series_uid: str
    series_time: str
    te: float
    ti: float
    tr_siemens: float
    fa: float
    tr: float
    seq_type: str
    diffusion: bool
    diffusion_dim: str
    diffusion_scheme: str
    accession_number: str
    body_part: str
    ref_physician: str
    study_date: str
    study_description: str
    study_id: str
    study_uid: str
    study_time: str

    @classmethod
    def from_ismrmrd_header(
        cls,
        header: ismrmrdschema.ismrmrdHeader,
        num_acq: int,
    ) -> KHeader:
        if len(header.encoding) != 1:
            raise NotImplementedError('Multiple encodings not supported')

        enc: ismrmrdschema.encodingType = header.encoding[0]
        encoding_limits = EncodingLimits.from_ismrmrd_encodingLimitsType(
            enc.encodingLimits)

        instance = cls(
            echo_train_length=enc.echoTrainLength,
            recon_fov=return_par_matrix_tensor(enc.reconSpace.fieldOfView_mm),
            recon_matrix=return_par_matrix_tensor(enc.reconSpace.matrixSize),
            acc_factor=return_acc_factor_tensor(
                enc.parallelImaging.accelerationFactor),
            calib_mode=enc.parallelImaging.calibrationMode.value,
            interleave_dim=enc.parallelImaging.interleavingDimension.value,
            multiband=enc.parallelImaging.multiband,
            traj_type=enc.trajectory.value,
            traj_description=enc.trajectoryDescription,
            encoding_fov=return_par_matrix_tensor(
                enc.encodedSpace.fieldOfView_mm),
            encoding_matrix=return_par_matrix_tensor(
                enc.encodedSpace.matrixSize),
            encoding_limits=encoding_limits,
            vendor=header.acquisitionSystemInformation.systemVendor,
            model=header.acquisitionSystemInformation.systemModel,
            station_name=header.acquisitionSystemInformation.stationName,
            device_id=header.acquisitionSystemInformation.deviceID,
            device_sn=header.acquisitionSystemInformation.deviceSerialNumber,
            coil_label=return_coil_label_dict(
                header.acquisitionSystemInformation.coilLabel),
            b0=header.acquisitionSystemInformation.systemFieldStrength_T,
            receiver_noise_bwdth=(
                header.acquisitionSystemInformation.relativeReceiverNoiseBandwidth),
            institution=header.acquisitionSystemInformation.institutionName,
            num_coils=header.acquisitionSystemInformation.receiverChannels,
            h1freq=header.experimentalConditions.H1resonanceFrequency_Hz,
            ref_frame_uid=header.measurementInformation.frameOfReferenceUID,
            series_num=header.measurementInformation.initialSeriesNumber,
            meas_depend=header.measurementInformation.measurementDependency,
            meas_id=header.measurementInformation.measurementID,
            pat_pos=header.measurementInformation.patientPosition.value,
            protocol_name=header.measurementInformation.protocolName,
            ref_im_seq=header.measurementInformation.referencedImageSequence,
            rel_table_pos=(
                header.measurementInformation.relativeTablePosition),
            seq_name=header.measurementInformation.sequenceName,
            series_date=header.measurementInformation.seriesDate,
            series_description=(
                header.measurementInformation.seriesDescription),
            series_uid=header.measurementInformation.seriesInstanceUIDRoot,
            series_time=header.measurementInformation.seriesTime,
            te=header.sequenceParameters.TE,
            ti=header.sequenceParameters.TI,
            tr_siemens=header.sequenceParameters.TR,
            fa=header.sequenceParameters.flipAngle_deg,
            tr=header.sequenceParameters.echo_spacing,
            seq_type=header.sequenceParameters.sequence_type,
            diffusion=header.sequenceParameters.diffusion,
            diffusion_dim=header.sequenceParameters.diffusionDimension,
            diffusion_scheme=header.sequenceParameters.diffusionScheme,
            accession_number=header.studyInformation.accessionNumber,
            body_part=header.studyInformation.bodyPartExamined,
            ref_physician=header.studyInformation.referringPhysicianName,
            study_date=header.studyInformation.studyDate,
            study_description=header.studyInformation.studyDescription,
            study_id=header.studyInformation.studyID,
            study_uid=header.studyInformation.studyInstanceUID,
            study_time=header.studyInformation.studyTime,
            acq_info=AcqInfo(num_acq),
        )
        return instance
