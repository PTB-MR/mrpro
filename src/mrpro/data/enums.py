"""All acquisition enums."""

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

from enum import Enum
from enum import Flag
from enum import auto


class AcqFlags(Flag):
    """Acquisition flags.

    Reference:
    https://github.com/ismrmrd/ismrmrd/blob/master/include/ismrmrd/ismrmrd.h
    NOTE: values in enum ISMRMRD_AcquisitionFlags start at 1 and not 0, but
    1 << (val-1) is used in 'ismrmrd_is_flag_set' function to calc bitmask value.
    """

    ACQ_NO_FLAG = 0
    ACQ_FIRST_IN_ENCODE_STEP1 = auto()
    ACQ_LAST_IN_ENCODE_STEP1 = auto()
    ACQ_FIRST_IN_ENCODE_STEP2 = auto()
    ACQ_LAST_IN_ENCODE_STEP2 = auto()
    ACQ_FIRST_IN_AVERAGE = auto()
    ACQ_LAST_IN_AVERAGE = auto()
    ACQ_FIRST_IN_SLICE = auto()
    ACQ_LAST_IN_SLICE = auto()
    ACQ_FIRST_IN_CONTRAST = auto()
    ACQ_LAST_IN_CONTRAST = auto()
    ACQ_FIRST_IN_PHASE = auto()
    ACQ_LAST_IN_PHASE = auto()
    ACQ_FIRST_IN_REPETITION = auto()
    ACQ_LAST_IN_REPETITION = auto()
    ACQ_FIRST_IN_SET = auto()
    ACQ_LAST_IN_SET = auto()
    ACQ_FIRST_IN_SEGMENT = auto()
    ACQ_LAST_IN_SEGMENT = auto()
    ACQ_IS_NOISE_MEASUREMENT = auto()
    ACQ_IS_PARALLEL_CALIBRATION = auto()
    ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING = auto()
    ACQ_IS_REVERSE = auto()
    ACQ_IS_NAVIGATION_DATA = auto()
    ACQ_IS_PHASECORR_DATA = auto()
    ACQ_LAST_IN_MEASUREMENT = auto()
    ACQ_IS_HPFEEDBACK_DATA = auto()
    ACQ_IS_DUMMYSCAN_DATA = auto()
    ACQ_IS_RTFEEDBACK_DATA = auto()
    ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA = auto()
    ACQ_IS_PHASE_STABILIZATION_REFERENCE = auto()
    ACQ_IS_PHASE_STABILIZATION = 30 << 1
    ACQ_COMPRESSION1 = 52 << 1  # 53 (on purpose!)
    ACQ_COMPRESSION2 = auto()
    ACQ_COMPRESSION3 = auto()
    ACQ_COMPRESSION4 = auto()
    ACQ_USER1 = auto()
    ACQ_USER2 = auto()
    ACQ_USER3 = auto()
    ACQ_USER4 = auto()
    ACQ_USER5 = auto()
    ACQ_USER6 = auto()
    ACQ_USER7 = auto()
    ACQ_USER8 = auto()


class InterleavingDimension(Enum):
    PHASE = 'phase'
    REPETITION = 'repetition'
    CONTRAST = 'contrast'
    AVERAGE = 'average'
    OTHER = 'other'


class MultibandCalibration(Enum):
    SEPARABLE2_D = 'separable2D'
    FULL3_D = 'full3D'
    OTHER = 'other'


class PatientPosition(Enum):
    HFP = 'HFP'
    HFS = 'HFS'
    HFDR = 'HFDR'
    HFDL = 'HFDL'
    FFP = 'FFP'
    FFS = 'FFS'
    FFDR = 'FFDR'
    FFDL = 'FFDL'
    OTHER = 'OTHER'


class TrajectoryType(Enum):
    CARTESIAN = 'cartesian'
    EPI = 'epi'
    RADIAL = 'radial'
    GOLDENANGLE = 'goldenangle'
    SPIRAL = 'spiral'
    OTHER = 'other'


class WaveformInformation(Enum):
    ECG = 'ecg'
    PULSE = 'pulse'
    RESPIRATORY = 'respiratory'
    TRIGGER = 'trigger'
    GRADIENTWAVEFORM = 'gradientwaveform'
    OTHER = 'other'


class CalibrationMode(Enum):
    EMBEDDED = 'embedded'
    INTERLEAVED = 'interleaved'
    SEPARATE = 'separate'
    EXTERNAL = 'external'
    OTHER = 'other'


class TrajType(Flag):
    SINGLEVALUE = 1
    ONGRID = 2
