"""Create ismrmrd datasets."""

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

from pathlib import Path
from typing import Literal

import ismrmrd
import ismrmrd.xsd
import numpy as np
import torch

from mrpro.phantoms import EllipsePhantom

ISMRMRD_TRAJECTORY_TYPE = (
    'cartesian',
    'epi',
    'radial',
    'goldenangle',
    'spiral',
    'other',
)


class IsmrmrdRawTestData:
    """Raw data in ISMRMRD format for testing.

    This is based on
    https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/generate_cartesian_shepp_logan_dataset.py

    Parameters
    ----------
    filename
        full path and filename
    matrix_size
        size of image matrix, by default 256
    ncoils
        number of coils, by default 8
    oversampling
        oversampling along readout (kx) direction, by default 2
    repetitions
        number of repetitions, by default 1
    flag_invalid_reps
        flag to indicate that number of phase encoding steps are different for repetitions, by default False
    acceleration
        undersampling along phase encoding (ky), by default 1
    noise_level
        scaling factor for noise level, by default 0.00005
    trajectory_type
        cartesian, by default cartesian
    sampling_order
        order how phase encoding points (ky) are obtained, by default linear
    phantom
        phantom with different ellipses
    """

    def __init__(
        self,
        filename: str | Path,
        matrix_size: int = 256,
        ncoils: int = 8,
        oversampling: int = 2,
        repetitions: int = 1,
        flag_invalid_reps: bool = False,
        acceleration: int = 1,
        noise_level: float = 0.00005,
        trajectory_type: str = 'cartesian',
        sampling_order: Literal['linear', 'low_high', 'high_low'] = 'linear',
        phantom: EllipsePhantom = EllipsePhantom(),
    ):
        self.filename: str | Path = filename
        self.matrix_size: int = matrix_size
        self.ncoils: int = ncoils
        self.oversampling: int = oversampling
        self.repetitions: int = repetitions
        self.flag_invalid_reps: bool = flag_invalid_reps
        self.acceleration: int = acceleration
        self.noise_level: float = noise_level
        self.trajectory_type: str = trajectory_type
        self.sampling_order: Literal['linear', 'low_high', 'high_low'] = sampling_order
        self.phantom: EllipsePhantom = phantom
        self.imref: torch.Tensor

        # The number of points in x,y,kx,ky
        nx = self.matrix_size
        ny = self.matrix_size
        nkx = self.oversampling * nx
        nky = ny

        # Create Cartesian grid for k-space locations
        ky_idx = self._calc_phase_encoding_steps(nky, self.acceleration, self.sampling_order)
        kx_idx = range(-nkx // 2, nkx // 2)
        [kx, ky] = np.meshgrid(kx_idx, ky_idx)

        # Create analytic k-space and reference image
        ktrue = self.phantom.kspace(torch.Tensor(ky), torch.Tensor(kx)).numpy()
        self.imref = self.phantom.image_space(nky, nkx)

        # Multi-coil acquisition
        # TODO: proper application of coils
        ktrue = np.tile(ktrue[None, ...], (self.ncoils, 1, 1))

        # Open the dataset
        dset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=True)

        # Create the XML header and write it to the file
        header = ismrmrd.xsd.ismrmrdHeader()

        # Experimental Conditions
        exp = ismrmrd.xsd.experimentalConditionsType()
        exp.H1resonanceFrequency_Hz = 128000000
        header.experimentalConditions = exp

        # Acquisition System Information
        sys = ismrmrd.xsd.acquisitionSystemInformationType()
        sys.receiverChannels = self.ncoils
        header.acquisitionSystemInformation = sys

        # Sequence Information
        seq = ismrmrd.xsd.sequenceParametersType()
        seq.TR = [89.6]
        seq.TE = [2.3]
        seq.TI = [0.0]
        seq.flipAngle_deg = 12.0
        seq.echo_spacing = 5.6
        header.sequenceParameters = seq

        # Encoding
        encoding = ismrmrd.xsd.encodingType()
        if self.trajectory_type in ISMRMRD_TRAJECTORY_TYPE:
            encoding.trajectory = ismrmrd.xsd.trajectoryType(self.trajectory_type)
        else:
            encoding.trajectory = ismrmrd.xsd.trajectoryType('other')

        # Encoded and recon spaces
        efov = ismrmrd.xsd.fieldOfViewMm()
        efov.x = self.oversampling * 256
        efov.y = 256
        efov.z = 5
        rfov = ismrmrd.xsd.fieldOfViewMm()
        rfov.x = 256
        rfov.y = 256
        rfov.z = 5

        ematrix = ismrmrd.xsd.matrixSizeType()
        ematrix.x = nkx
        ematrix.y = nky
        ematrix.z = 1
        rmatrix = ismrmrd.xsd.matrixSizeType()
        rmatrix.x = nx
        rmatrix.y = ny
        rmatrix.z = 1

        espace = ismrmrd.xsd.encodingSpaceType()
        espace.matrixSize = ematrix
        espace.fieldOfView_mm = efov
        rspace = ismrmrd.xsd.encodingSpaceType()
        rspace.matrixSize = rmatrix
        rspace.fieldOfView_mm = rfov

        # Set encoded and recon spaces
        encoding.encodedSpace = espace
        encoding.reconSpace = rspace

        # Encoding limits
        limits = ismrmrd.xsd.encodingLimitsType()

        limits1 = ismrmrd.xsd.limitType()
        limits1.minimum = 0
        limits1.center = ny // 2
        limits1.maximum = ny - 1
        limits.kspace_encoding_step_1 = limits1

        limits_rep = ismrmrd.xsd.limitType()
        limits_rep.minimum = 0
        limits_rep.center = self.repetitions // 2
        limits_rep.maximum = self.repetitions - 1
        limits.repetition = limits_rep

        encoding.encodingLimits = limits
        header.encoding.append(encoding)

        dset.write_xml_header(header.toXML('utf-8'))

        # Create an acquistion and reuse it
        acq = ismrmrd.Acquisition()
        acq.resize(nkx, self.ncoils)
        acq.version = 1
        acq.available_channels = self.ncoils
        acq.center_sample = round(nkx / 2)
        acq.read_dir[0] = 1.0
        acq.phase_dir[1] = 1.0
        acq.slice_dir[2] = 1.0

        # Initialize an acquisition counter
        counter = 0

        # Write out a few noise scans
        for n in range(32):
            noise = self.noise_level * (np.random.randn(self.ncoils, nkx) + 1j * np.random.randn(self.ncoils, nkx))
            # here's where we would make the noise correlated
            acq.scan_counter = counter
            acq.clearAllFlags()
            acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            acq.data[:] = noise
            dset.append_acquisition(acq)
            counter += 1  # increment the scan counter

        # Loop over the repetitions, add noise and write to disk
        for rep in range(self.repetitions):
            noise = self.noise_level * (
                np.random.randn(self.ncoils, nky // self.acceleration, nkx)
                + 1j * np.random.randn(self.ncoils, nky // self.acceleration, nkx)
            )
            # Here's where we would make the noise correlated
            K = ktrue + noise
            acq.idx.repetition = rep
            for idx, line in enumerate(ky_idx):
                if not self.flag_invalid_reps or rep == 0 or idx < len(ky_idx) // 2:  # fewer lines for rep > 0
                    # Set some fields in the header
                    line_idx = line + nky // 2
                    acq.scan_counter = counter
                    acq.idx.kspace_encode_step_1 = line_idx
                    acq.clearAllFlags()
                    if line == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif line == nky - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
                    # Set the data and append
                    acq.data[:] = K[:, idx, :]
                    dset.append_acquisition(acq)
                    counter += 1

        # Clean up
        dset.close()

    @staticmethod
    def _calc_phase_encoding_steps(
        nky: int,
        acceleration: int = 1,
        sampling_order: Literal['linear', 'low_high', 'high_low'] = 'linear',
    ):
        """Calculate nky phase encoding points.

        Parameters
        ----------
        nky
            number of phase encoding points before undersampling
        acceleration, optional
            undersampling factor, by default 1
        sampling_order, optional
            order how phase encoding points are sampled, by default "linear"
        """
        # Always include k-space center and more points on the negative side of k-space
        ky_pos = np.arange(0, nky // 2, acceleration)
        ky_neg = -np.arange(acceleration, nky // 2 + 1, acceleration)
        ky = np.concatenate((ky_neg, ky_pos), axis=0)

        if sampling_order == 'linear':
            ky = np.sort(ky)
        elif sampling_order == 'low_high':
            idx = np.argsort(np.abs(ky), kind='stable')
            ky = ky[idx]
        elif sampling_order == 'high_low':
            idx = np.argsort(-np.abs(ky), kind='stable')
            ky = ky[idx]
        else:
            raise ValueError(f'sampling order {sampling_order} not supported.')
        return ky
