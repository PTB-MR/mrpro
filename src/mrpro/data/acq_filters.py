"""Test ISMRMRD acquisitions based on their flags."""

import ismrmrd

from mrpro.data.enums import AcqFlags

# Same criteria as https://github.com/wtclarke/pymapvbvd/blob/master/mapvbvd/mapVBVD.py uses
DEFAULT_IGNORE_FLAGS = (
    AcqFlags.ACQ_IS_NOISE_MEASUREMENT
    | AcqFlags.ACQ_IS_DUMMYSCAN_DATA
    | AcqFlags.ACQ_IS_HPFEEDBACK_DATA
    | AcqFlags.ACQ_IS_NAVIGATION_DATA
    | AcqFlags.ACQ_IS_PHASECORR_DATA
    | AcqFlags.ACQ_IS_PHASE_STABILIZATION
    | AcqFlags.ACQ_IS_PHASE_STABILIZATION_REFERENCE
    | AcqFlags.ACQ_IS_PARALLEL_CALIBRATION
)


def is_image_acquisition(acquisition: ismrmrd.Acquisition) -> bool:
    """Test if acquisition was obtained for imaging.

    Parameters
    ----------
    acquisition
        ISMRMRD acquisition

    Returns
    -------
        True if the acquisition was obtained for imaging
    """
    return not DEFAULT_IGNORE_FLAGS.value & acquisition.flags


def is_noise_acquisition(acquisition: ismrmrd.Acquisition) -> bool:
    """Test if acquisition contains noise measurements.

    Parameters
    ----------
    acquisition
        ISMRMRD acquisition

    Returns
    -------
        True if the acquisition contains a noise measurement
    """
    return AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value & acquisition.flags


def is_coil_calibration_acquisition(acquisition: ismrmrd.Acquisition) -> bool:
    """Test if acquisitions was obtained for coil calibration.

    Parameters
    ----------
    acquisition
        ISMRMRD acquisition

    Returns
    -------
        True if the acquisition contains coil calibration data
    """
    coil_calibration_flag = AcqFlags.ACQ_IS_PARALLEL_CALIBRATION | AcqFlags.ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING
    return coil_calibration_flag.value & acquisition.flags


def has_n_coils(n_coils: int, acquisition: ismrmrd.Acquisition) -> bool:
    """Test if acquisitions was obtained with a certain number of receiver coils.

    Parameters
    ----------
    n_coils
        number of receiver coils
    acquisition
        ISMRMRD acquisition

    Returns
    -------
        True if the acquisition was obtained with n_coils receiver coils
    """
    return acquisition.data.shape[0] == n_coils
