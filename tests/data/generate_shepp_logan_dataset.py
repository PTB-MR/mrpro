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

import os

import ismrmrd
import ismrmrd.xsd
import numpy as np
import scipy.special as sp_special


def create_analytic_2d_kspace(ky: np.ndarray, kx: np.ndarray):
    """Create 2D analytic kspace data based on given k-space locations.

    Parameters
    ----------
    ky
        k-space locations in ky
    kx
        k-space loations in kx. Same shape as ky.
    """
    # kx and ky have to be of same shape
    if kx.shape != ky.shape:
        raise ValueError(f'shape mismatch between kx {kx.shape} and ky {ky.shape}')

    # Create k-space data for three circles of different intensity
    par = [[0.2, 0.2, 0.2, 0.2], [0.1, -0.1, 0.2, 0.2], [-0.2, 0.1, 0.3, 0.3]]
    intensity = [1, 2, 4]

    kdat = 0
    for ind, ipar in enumerate(par):
        arg = np.sqrt(ipar[2] ** 2 * kx ** 2 + ipar[3] ** 2 * ky ** 2)
        arg[arg < 1e-6] = 1e-6  # avoid zeros

        cdat = ipar[2] * ipar[3] * 0.5 * sp_special.jv(1, np.pi * arg) / arg
        kdat += (np.exp(1j * 2 * np.pi * (ipar[0] * kx + ipar[1] * ky))
                 * cdat * intensity[ind]).astype(np.complex64)

    return (kdat)


def create(filename: str | os.PathLike, matrix_size: int = 256, ncoils: int = 8, oversampling: int = 2,
           repetitions: int = 1, acceleration: int = 1, noise_level: float = 0.05):
    """Create ismrmrd raw data file.

    This is based on
    https://github.com/ismrmrd/ismrmrd-python-tools/blob/master/generate_cartesian_shepp_logan_dataset.py

    Parameters
    ----------
    filename, optional
        full path and filename
    matrix_size, optional
        size of image matrix, by default 256
    ncoils, optional
        number of coils, by default 8
    oversampling, optional
        oversampling along readout (kx) direction, by default 2
    repetitions, optional
        number of repetitions, by default 1
    acceleration, optional
        undersampling along phase encoding (ky), by default 1
    noise_level, optional
        scaling factor for noise level, by default 0.05
    """
    print('Creating Shepp-Logan phantom raw data:')
    print('Matrix %dx%d at R=%d with %d repetitions' % (matrix_size, matrix_size, acceleration, repetitions))
    print('%d ncoils with %d oversampling and %1.2f noise level' % (ncoils, oversampling, noise_level))

    # The number of points in x,y,kx,ky
    nx = matrix_size
    ny = matrix_size
    nkx = oversampling*nx
    nky = ny

    # Create Cartesian grid for k-space locations
    ky_idx = range(-nky//2, nky//2)  # TODO: take acceleration into account
    kx_idx = range(-nkx//2, nkx//2)
    [kx, ky] = np.meshgrid(kx_idx, ky_idx)
    ktrue = create_analytic_2d_kspace(ky, kx)

    # Multi-coil acquisition
    # TODO: proper application of coils
    ktrue = np.tile(ktrue[None, ...], (ncoils, 1, 1))

    # Open the dataset
    dset = ismrmrd.Dataset(filename, 'dataset', create_if_needed=True)

    # Create the XML header and write it to the file
    header = ismrmrd.xsd.ismrmrdHeader()

    # Experimental Conditions
    exp = ismrmrd.xsd.experimentalConditionsType()
    exp.H1resonanceFrequency_Hz = 128000000
    header.experimentalConditions = exp

    # Acquisition System Information
    sys = ismrmrd.xsd.acquisitionSystemInformationType()
    sys.receiverChannels = ncoils
    header.acquisitionSystemInformation = sys

    # Sequence Information
    seq = ismrmrd.xsd.sequenceParametersType()
    seq.TR = [89.6,]
    seq.TE = [2.3,]
    seq.TI = [0.0,]
    seq.flipAngle_deg = 12.0
    seq.echo_spacing = 5.6
    header.sequenceParameters = seq

    # Encoding
    encoding = ismrmrd.xsd.encodingType()
    encoding.trajectory = ismrmrd.xsd.trajectoryType('cartesian')

    # encoded and recon spaces
    efov = ismrmrd.xsd.fieldOfViewMm()
    efov.x = oversampling*256
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
    limits1.center = round(ny/2)
    limits1.maximum = ny - 1
    limits.kspace_encoding_step_1 = limits1

    limits_rep = ismrmrd.xsd.limitType()
    limits_rep.minimum = 0
    limits_rep.center = round(repetitions / 2)
    limits_rep.maximum = repetitions - 1
    limits.repetition = limits_rep

    limits_rest = ismrmrd.xsd.limitType()
    limits_rest.minimum = 0
    limits_rest.center = 0
    limits_rest.maximum = 0
    limits.kspace_encoding_step_0 = limits_rest
    limits.slice = limits_rest
    limits.average = limits_rest
    limits.contrast = limits_rest
    limits.kspaceEncodingStep2 = limits_rest
    limits.phase = limits_rest
    limits.segment = limits_rest
    limits.set = limits_rest

    encoding.encodingLimits = limits
    header.encoding.append(encoding)

    dset.write_xml_header(header.toXML('utf-8'))

    # Create an acquistion and reuse it
    acq = ismrmrd.Acquisition()
    acq.resize(nkx, ncoils)
    acq.version = 1
    acq.available_channels = ncoils
    acq.center_sample = round(nkx/2)
    acq.read_dir[0] = 1.0
    acq.phase_dir[1] = 1.0
    acq.slice_dir[2] = 1.0

    # Initialize an acquisition counter
    counter = 0

    # Write out a few noise scans
    for n in range(32):
        noise = noise_level * (np.random.randn(ncoils, nkx) + 1j * np.random.randn(ncoils, nkx))
        # here's where we would make the noise correlated
        acq.scan_counter = counter
        acq.clearAllFlags()
        acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        acq.data[:] = noise
        dset.append_acquisition(acq)
        counter += 1  # increment the scan counter

    # Loop over the repetitions, add noise and write to disk
    # simulating a T-SENSE type scan
    for rep in range(repetitions):
        noise = noise_level * (np.random.randn(ncoils, nky, nkx) + 1j * np.random.randn(ncoils, nky, nkx))
        # Here's where we would make the noise correlated
        K = ktrue + noise
        acq.idx.repetition = rep
        for line in ky_idx:
            # Set some fields in the header
            line_idx = line + nky//2
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
            # set the data and append
            acq.data[:] = K[:, line_idx, :]
            dset.append_acquisition(acq)
            counter += 1

    # Clean up
    dset.close()
