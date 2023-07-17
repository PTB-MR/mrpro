"""Data classes for MR raw data."""

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

import os
import re
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Dict
from typing import List

import ismrmrd
import numpy as np
import torch

from mrpro.data.traj import KTrajectory

# Acquisition flags labelling each readout. Readouts can have multiple flags.
ACQ_FLAGS = ('ACQ_NO_FLAG',
             'ACQ_FIRST_IN_ENCODE_STEP1',
             'ACQ_LAST_IN_ENCODE_STEP1',
             'ACQ_FIRST_IN_ENCODE_STEP2',
             'ACQ_LAST_IN_ENCODE_STEP2',
             'ACQ_FIRST_IN_AVERAGE',
             'ACQ_LAST_IN_AVERAGE',
             'ACQ_FIRST_IN_SLICE',
             'ACQ_LAST_IN_SLICE',
             'ACQ_FIRST_IN_CONTRAST',
             'ACQ_LAST_IN_CONTRAST',
             'ACQ_FIRST_IN_PHASE',
             'ACQ_LAST_IN_PHASE',
             'ACQ_FIRST_IN_REPETITION',
             'ACQ_LAST_IN_REPETITION',
             'ACQ_FIRST_IN_SET',
             'ACQ_LAST_IN_SET',
             'ACQ_FIRST_IN_SEGMENT',
             'ACQ_LAST_IN_SEGMENT',
             'ACQ_IS_NOISE_MEASUREMENT',
             'ACQ_IS_PARALLEL_CALIBRATION',
             'ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING',
             'ACQ_IS_REVERSE',
             'ACQ_IS_NAVIGATION_DATA',
             'ACQ_IS_PHASECORR_DATA',
             'ACQ_LAST_IN_MEASUREMENT',
             'ACQ_IS_HPFEEDBACK_DATA',
             'ACQ_IS_DUMMYSCAN_DATA',
             'ACQ_IS_RTFEEDBACK_DATA',
             'ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA',
             'ACQ_USER1',
             'ACQ_USER2',
             'ACQ_USER3',
             'ACQ_USER4',
             'ACQ_USER5',
             'ACQ_USER6',
             'ACQ_USER7',
             'ACQ_USER8')


class ELimits():
    """Encoding limits Each encoding limit is described as [min, max,
    center]"""
    __slots__ = ('kspace_encoding_step_1',
                 'kspace_encoding_step_2',
                 'average',
                 'slice',
                 'contrast',
                 'phase',
                 'repetition',
                 'set',
                 'segment')

    def __init__(self) -> None:
        pass


class AcqInfo():
    __slots__ = ('acquisition_time_stamp',
                 'active_channels',
                 'available_channels',
                 'average',
                 'center_sample',
                 'channel_mask',
                 'contrast',
                 'discard_post',
                 'discard_pre',
                 'encoding_space_ref',
                 'flags',
                 'kspace_encode_step_1',
                 'kspace_encode_step_2',
                 'measurement_uid',
                 'number_of_samples',
                 'patient_table_position',
                 'phase',
                 'phase_dir',
                 'physiology_time_stamp',
                 'position',
                 'read_dir',
                 'repetition',
                 'sample_time_us',
                 'scan_counter',
                 'segment',
                 'set',
                 'slice',
                 'slice_dir',
                 'trajectory_dimensions',
                 'user_float',
                 'user_int',
                 'version',
                 )

    def __init__(self, num_acq: int) -> None:
        self.acquisition_time_stamp = torch.zeros(
            (num_acq,), dtype=torch.int64)
        self.active_channels = torch.zeros((num_acq,), dtype=torch.int64)
        self.available_channels = torch.zeros((num_acq,), dtype=torch.int64)
        self.average = torch.zeros((num_acq,), dtype=torch.int64)
        self.center_sample = torch.zeros((num_acq,), dtype=torch.int64)
        self.channel_mask = torch.zeros(
            (num_acq, ismrmrd.constants.CHANNEL_MASKS), dtype=torch.int64)
        self.contrast = torch.zeros((num_acq,), dtype=torch.int64)
        self.discard_post = torch.zeros((num_acq,), dtype=torch.int64)
        self.discard_pre = torch.zeros((num_acq,), dtype=torch.int64)
        self.encoding_space_ref = torch.zeros((num_acq,), dtype=torch.int64)
        self.flags = torch.zeros((num_acq,), dtype=torch.int64)
        self.kspace_encode_step_1 = torch.zeros((num_acq,), dtype=torch.int64)
        self.kspace_encode_step_2 = torch.zeros((num_acq,), dtype=torch.int64)
        self.measurement_uid = torch.zeros((num_acq,), dtype=torch.int64)
        self.number_of_samples = torch.zeros((num_acq,), dtype=torch.int64)
        self.patient_table_position = torch.zeros(
            (num_acq, ismrmrd.constants.POSITION_LENGTH), dtype=torch.float32)
        self.phase = torch.zeros((num_acq,), dtype=torch.int64)
        self.phase_dir = torch.zeros(
            (num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.physiology_time_stamp = torch.zeros(
            (num_acq, ismrmrd.constants.PHYS_STAMPS), dtype=torch.float32)
        self.position = torch.zeros(
            (num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.read_dir = torch.zeros(
            (num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.repetition = torch.zeros((num_acq,), dtype=torch.int64)
        self.sample_time_us = torch.zeros((num_acq,), dtype=torch.int64)
        self.scan_counter = torch.zeros((num_acq,), dtype=torch.int64)
        self.segment = torch.zeros((num_acq,), dtype=torch.int64)
        self.set = torch.zeros((num_acq,), dtype=torch.int64)
        self.slice = torch.zeros((num_acq,), dtype=torch.int64)
        self.slice_dir = torch.zeros(
            (num_acq, ismrmrd.constants.DIRECTION_LENGTH), dtype=torch.float32)
        self.trajectory_dimensions = torch.zeros((num_acq,), dtype=torch.int64)
        self.user_float = torch.zeros(
            (num_acq, ismrmrd.constants.USER_FLOATS), dtype=torch.float32)
        self.user_int = torch.zeros(
            (num_acq, ismrmrd.constants.USER_INTS), dtype=torch.int64)
        self.version = torch.zeros((num_acq,), dtype=torch.int64)

    def from_ismrmrd_acq_header(self, curr_idx: int, acq: ismrmrd.Acquisition) -> None:
        for slot in self.__slots__:
            curr_attr = getattr(self, slot)
            if slot in ('kspace_encode_step_1', 'kspace_encode_step_2', 'average', 'slice',
                        'contrast', 'phase', 'repetition', 'set', 'segment'):
                curr_attr[curr_idx, ...] = torch.tensor(
                    getattr(acq.idx, slot), dtype=curr_attr.dtype)

            else:
                curr_attr[curr_idx, ...] = torch.tensor(
                    getattr(acq, slot), dtype=curr_attr.dtype)
            setattr(self, slot, curr_attr)


def _return_par_tensor(par, array_attr) -> torch.Tensor | None:
    if par is None:
        return None
    else:
        par_tensor = []
        for attr in array_attr:
            par_tensor.append(getattr(par, attr))
        return (torch.tensor(par_tensor))


def return_par_matrix_tensor(par: ismrmrd.xsd.ismrmrdschema.ismrmrd.matrixSizeType) -> torch.Tensor | None:
    return (_return_par_tensor(par, array_attr=('x', 'y', 'z')))


def return_par_enc_limits_tensor(par: ismrmrd.xsd.ismrmrdschema.ismrmrd.limitType) -> torch.Tensor | None:
    return (_return_par_tensor(par, array_attr=('minimum', 'maximum', 'center')))


def return_acc_factor_tensor(par: ismrmrd.xsd.ismrmrdschema.ismrmrd.accelerationFactorType) -> torch.Tensor | None:
    return (_return_par_tensor(par, array_attr=('kspace_encoding_step_1', 'kspace_encoding_step_2')))


def bitmask_flag_to_strings(flag: int):
    if flag > 0:
        bmask = '{0:064b}'.format(flag)
        bitmask_idx = [m.start() + 1 for m in re.finditer('1', bmask[::-1])]
    else:
        bitmask_idx = [0, ]
    flag_strings = []
    for knd in range(len(bitmask_idx)):
        flag_strings.append(ACQ_FLAGS[bitmask_idx[knd]])
    return (flag_strings)


def return_coil_label_dict(coil_label: List[ismrmrd.xsd.ismrmrdschema.ismrmrd.coilLabelType]) -> Dict:
    coil_label_dict = {}
    for idx, label in enumerate(coil_label):
        coil_label_dict[idx] = [label.coilNumber, label.coilName]
    return (coil_label_dict)


class KHeader():
    __slots__ = ('protocol_name', 'pat_pos', 'meas_id', 'institution', 'receiver_noise_bwdth', 'b0', 'model', 'vendor',
                 'elimits', 'acc_factor', 'rec_matrix', 'rec_fov', 'enc_matrix', 'enc_fov', 'etl', 'num_coils', 'acq_info', 'calib_mode',
                 'interleave_dim', 'multiband', 'traj_type', 'traj_description', 'device_id', 'device_sn', 'coil_label', 'station_name',
                 'h1freq', 'ref_frame_uid', 'series_num', 'meas_depend', 'ref_im_seq', 'rel_table_pos', 'seq_name', 'series_date',
                 'series_description', 'series_uid', 'series_time', 'te', 'ti', 'tr_siemens', 'fa', 'tr', 'seq_type', 'diffusion',
                 'diffusion_dim', 'diffusion_scheme', 'accession_number', 'body_part', 'ref_physician', 'study_date', 'study_description',
                 'study_id', 'study_uid', 'study_time')

    def __init__(self) -> None:
        pass

    def from_ismrmrd_header(self, header: ismrmrd.xsd.ismrmrdschema.ismrmrd.ismrmrdHeader, num_acq: int) -> None:

        # Encoding
        assert len(header.encoding) == 1, 'Multiple encodings are not supported.'
        enc = header.encoding[0]
        self.etl = enc.echoTrainLength
        self.enc_fov = return_par_matrix_tensor(
            enc.encodedSpace.fieldOfView_mm)
        self.enc_matrix = return_par_matrix_tensor(enc.encodedSpace.matrixSize)
        self.rec_fov = return_par_matrix_tensor(enc.reconSpace.fieldOfView_mm)
        self.rec_matrix = return_par_matrix_tensor(enc.reconSpace.matrixSize)
        self.acc_factor = return_acc_factor_tensor(
            enc.parallelImaging.accelerationFactor)
        self.calib_mode = enc.parallelImaging.calibrationMode.value
        self.interleave_dim = enc.parallelImaging.interleavingDimension.value
        self.multiband = enc.parallelImaging.multiband
        self.traj_type = enc.trajectory.value
        self.traj_description = enc.trajectoryDescription

        self.elimits = ELimits()
        for climit in self.elimits.__slots__:
            setattr(self.elimits, climit, return_par_enc_limits_tensor(
                getattr(enc.encodingLimits, climit)))

        # AcquisitionSystemInformation
        self.vendor = header.acquisitionSystemInformation.systemVendor
        self.model = header.acquisitionSystemInformation.systemModel
        self.station_name = header.acquisitionSystemInformation.stationName
        self.device_id = header.acquisitionSystemInformation.deviceID
        self.device_sn = header.acquisitionSystemInformation.deviceSerialNumber
        self.coil_label = return_coil_label_dict(
            header.acquisitionSystemInformation.coilLabel)
        self.b0 = header.acquisitionSystemInformation.systemFieldStrength_T
        self.receiver_noise_bwdth = header.acquisitionSystemInformation.relativeReceiverNoiseBandwidth
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
        self.rel_table_pos = header.measurementInformation.relativeTablePosition
        self.seq_name = header.measurementInformation.sequenceName
        self.series_date = header.measurementInformation.seriesDate
        self.series_description = header.measurementInformation.seriesDescription
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


class KData():
    def __init__(self,
                 header: KHeader,
                 data: torch.Tensor,
                 traj: torch.Tensor) -> None:
        self.header: KHeader = header
        self._data: torch.Tensor = data
        self._traj: torch.Tensor = traj

    @classmethod
    def from_file(cls,
                  filename: str | Path,
                  ktrajectory_calculator: KTrajectory) -> KData:

        # Check file is valid
        if not os.path.isfile(filename):
            print('%s is not a valid file' % filename)
            raise SystemExit

        # Read header
        dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=False)
        hdr_xml = dset.read_xml_header()
        hdr = ismrmrd.xsd.CreateFromDocument(hdr_xml)
        dset.close()

        # Read k-space data
        with ismrmrd.File(filename) as mrd:
            acqs = mrd['dataset'].acquisitions[:]

        # Get indices for imaging data
        im_idx = []
        unique_acq_flags = set()
        for idx, acq in enumerate(acqs):
            for el in bitmask_flag_to_strings(acq.flags):
                unique_acq_flags.add(el)
            if 'ACQ_IS_NOISE_MEASUREMENT' not in bitmask_flag_to_strings(acq.flags):
                im_idx.append(idx)

        kheader = KHeader()
        kheader.from_ismrmrd_header(hdr, len(im_idx))

        # Get k-space data
        kdata = torch.zeros((len(im_idx), kheader.num_coils,
                            acqs[im_idx[0]].number_of_samples), dtype=torch.complex64)
        for idx in range(len(im_idx)):
            acq = acqs[im_idx[idx]]
            kdata[idx, :, :] = torch.tensor(acq.data, dtype=torch.complex64)
            # TODO: Make this faster
            kheader.acq_info.from_ismrmrd_acq_header(idx, acq)

        # Calculate trajectory
        ktraj = ktrajectory_calculator.calc_traj(kheader)

        # TODO: Check for partial Fourier and reflected readouts

        # Sort k-space data into (dim4, ncoils, k2, k1, k0)
        kdim_labels = ('kspace_encode_step_1', 'kspace_encode_step_2',
                       'average', 'slice', 'contrast', 'phase', 'repetition', 'set')
        kdim_num = np.asarray(
            [len(np.unique(getattr(kheader.acq_info, acq_label))) for acq_label in kdim_labels])

        # Ensure each dim4 covers the same number of k2 and k1 points
        for idx, acq_label in enumerate(kdim_labels[2:]):
            label_values = np.unique(getattr(kheader.acq_info, acq_label))
            for ind in range(len(label_values)):
                cidx_curr_label = tuple(np.where(
                    getattr(kheader.acq_info, acq_label) == label_values[ind])[0])
                kdim_label_k1 = len(
                    np.unique(kheader.acq_info.kspace_encode_step_1[cidx_curr_label]))
                kdim_label_k2 = len(
                    np.unique(kheader.acq_info.kspace_encode_step_2[cidx_curr_label]))
                assert kdim_label_k1 == kdim_num[0], f'{acq_label} has {kdim_label_k1} k1 points instead of {kdim_num[0]}'
                assert kdim_label_k2 == kdim_num[1], f'{acq_label} has {kdim_label_k2} k2 points instead of {kdim_num[1]}'

        sort_ki = np.stack((kheader.acq_info.kspace_encode_step_1, kheader.acq_info.kspace_encode_step_2, kheader.acq_info.average,
                            kheader.acq_info.slice, kheader.acq_info.contrast, kheader.acq_info.phase,
                            kheader.acq_info.repetition, kheader.acq_info.set), axis=0)
        sort_idx = np.lexsort(sort_ki)

        new_shape = (np.prod(kdim_num[2:]), kdim_num[1], kdim_num[0],)
        kdata = torch.reshape(kdata[sort_idx, :, :],
                              new_shape + kdata.shape[1:])
        kdata = torch.moveaxis(kdata, (0, 1, 2, 3, 4), (0, 2, 3, 1, 4))

        ktraj = torch.reshape(ktraj[sort_idx, :, :],
                              new_shape + ktraj.shape[1:])
        ktraj = torch.moveaxis(ktraj, (0, 1, 2, 3, 4), (0, 2, 3, 1, 4))

        for slot in kheader.acq_info.__slots__:
            curr_attr = getattr(kheader.acq_info, slot)
            if curr_attr.ndim == 2:
                curr_shape = new_shape + (curr_attr.shape[1],)
            else:
                curr_shape = new_shape
            setattr(kheader.acq_info, slot, torch.reshape(
                curr_attr[sort_idx, ...], curr_shape))

        return cls(kheader, kdata, ktraj)

    @property
    def traj(self) -> torch.Tensor:
        return self._traj

    @traj.setter
    def traj(self, value: torch.Tensor):
        self._traj = value

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, value: torch.Tensor):
        self._data = value
