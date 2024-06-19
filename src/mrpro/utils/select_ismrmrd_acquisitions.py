"""Select ISMRMRD acquisitions based on their flags."""

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

def select_image_acquisition(acquisition: ismrmrd.Acquisition) -> bool:
    """Select acquisition obtained for imaging.

    Parameters
    ----------
    acquisition
        ISMRMRD acquisition

    Returns
    -------
        True if the acquisition was obtained for imaging
    """
    return not DEFAULT_IGNORE_FLAGS.value & acquisition.flags

def select_noise_acquisition(acquisition: ismrmrd.Acquisition) -> bool:
    """Select acquisition containing noise measurments.

    Parameters
    ----------
    acquisition
        ISMRMRD acquisition

    Returns
    -------
        True if the acquisition contains a noise measurement
    """
    return AcqFlags.ACQ_IS_NOISE_MEASUREMENT.value & acquisition.flags

def select_coil_calibration_acquisition(acquisition: ismrmrd.Acquisition) -> bool:
    """Select acquisition for coil calibration.

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
