"""All acquisition enums."""

import enum


class AcqFlags(enum.Flag):
    """Acquisition flags.

    NOTE: values in enum ISMRMRD_AcquisitionFlags start at 1 and not 0, but
    1 << (val-1) is used in 'ismrmrd_is_flag_set' function to calc bitmask value [ISMb]_.

    References
    ----------
    .. [ISMb] ISMRMRD https://github.com/ismrmrd/ismrmrd/blob/master/include/ismrmrd/ismrmrd.h
    """

    ACQ_NO_FLAG = 0
    ACQ_FIRST_IN_ENCODE_STEP1 = enum.auto()
    ACQ_LAST_IN_ENCODE_STEP1 = enum.auto()
    ACQ_FIRST_IN_ENCODE_STEP2 = enum.auto()
    ACQ_LAST_IN_ENCODE_STEP2 = enum.auto()
    ACQ_FIRST_IN_AVERAGE = enum.auto()
    ACQ_LAST_IN_AVERAGE = enum.auto()
    ACQ_FIRST_IN_SLICE = enum.auto()
    ACQ_LAST_IN_SLICE = enum.auto()
    ACQ_FIRST_IN_CONTRAST = enum.auto()
    ACQ_LAST_IN_CONTRAST = enum.auto()
    ACQ_FIRST_IN_PHASE = enum.auto()
    ACQ_LAST_IN_PHASE = enum.auto()
    ACQ_FIRST_IN_REPETITION = enum.auto()
    ACQ_LAST_IN_REPETITION = enum.auto()
    ACQ_FIRST_IN_SET = enum.auto()
    ACQ_LAST_IN_SET = enum.auto()
    ACQ_FIRST_IN_SEGMENT = enum.auto()
    ACQ_LAST_IN_SEGMENT = enum.auto()
    ACQ_IS_NOISE_MEASUREMENT = enum.auto()
    ACQ_IS_PARALLEL_CALIBRATION = enum.auto()
    ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING = enum.auto()
    ACQ_IS_REVERSE = enum.auto()
    ACQ_IS_NAVIGATION_DATA = enum.auto()
    ACQ_IS_PHASECORR_DATA = enum.auto()
    ACQ_LAST_IN_MEASUREMENT = enum.auto()
    ACQ_IS_HPFEEDBACK_DATA = enum.auto()
    ACQ_IS_DUMMYSCAN_DATA = enum.auto()
    ACQ_IS_RTFEEDBACK_DATA = enum.auto()
    ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA = enum.auto()
    ACQ_IS_PHASE_STABILIZATION_REFERENCE = enum.auto()
    ACQ_IS_PHASE_STABILIZATION = 30 << 1
    ACQ_COMPRESSION1 = 52 << 1  # 53 (on purpose!)
    ACQ_COMPRESSION2 = enum.auto()
    ACQ_COMPRESSION3 = enum.auto()
    ACQ_COMPRESSION4 = enum.auto()
    ACQ_USER1 = enum.auto()
    ACQ_USER2 = enum.auto()
    ACQ_USER3 = enum.auto()
    ACQ_USER4 = enum.auto()
    ACQ_USER5 = enum.auto()
    ACQ_USER6 = enum.auto()
    ACQ_USER7 = enum.auto()
    ACQ_USER8 = enum.auto()


class InterleavingDimension(enum.Enum):
    """Interleaving dimension."""

    PHASE = 'phase'
    REPETITION = 'repetition'
    CONTRAST = 'contrast'
    AVERAGE = 'average'
    OTHER = 'other'


class MultibandCalibration(enum.Enum):
    """Multiband calibration."""

    SEPARABLE2_D = 'separable2D'
    FULL3_D = 'full3D'
    OTHER = 'other'


class PatientPosition(enum.Enum):
    """Patient position."""

    HFP = 'HFP'
    HFS = 'HFS'
    HFDR = 'HFDR'
    HFDL = 'HFDL'
    FFP = 'FFP'
    FFS = 'FFS'
    FFDR = 'FFDR'
    FFDL = 'FFDL'
    OTHER = 'OTHER'


class TrajectoryType(enum.Enum):
    """Trajectory type."""

    CARTESIAN = 'cartesian'
    EPI = 'epi'
    RADIAL = 'radial'
    GOLDENANGLE = 'goldenangle'
    SPIRAL = 'spiral'
    OTHER = 'other'


class WaveformInformation(enum.Enum):
    """Waveform information."""

    ECG = 'ecg'
    PULSE = 'pulse'
    RESPIRATORY = 'respiratory'
    TRIGGER = 'trigger'
    GRADIENTWAVEFORM = 'gradientwaveform'
    OTHER = 'other'


class CalibrationMode(enum.Enum):
    """Calibration mode."""

    EMBEDDED = 'embedded'
    INTERLEAVED = 'interleaved'
    SEPARATE = 'separate'
    EXTERNAL = 'external'
    OTHER = 'other'


class TrajType(enum.Flag):
    """Special Properties of the Trajectory."""

    SINGLEVALUE = 1
    ONGRID = 2
