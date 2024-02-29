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

from __future__ import annotations

from pathlib import Path
from typing import Literal

import ismrmrd
import ismrmrd.xsd
import torch
from einops import repeat

from mrpro.data import SpatialDimension
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
        trajectory_type: Literal['cartesian', 'radial'] = 'cartesian',
        sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
        phantom: EllipsePhantom | None = None,
    ):
        if not phantom:
            phantom = EllipsePhantom()

        self.filename: str | Path = filename
        self.matrix_size: int = matrix_size
        self.ncoils: int = ncoils
        self.oversampling: int = oversampling
        self.repetitions: int = repetitions
        self.flag_invalid_reps: bool = flag_invalid_reps
        self.acceleration: int = acceleration
        self.noise_level: float = noise_level
        self.trajectory_type: Literal['cartesian', 'radial'] = trajectory_type
        self.sampling_order: Literal['linear', 'low_high', 'high_low'] = sampling_order
        self.phantom: EllipsePhantom = phantom
        self.imref: torch.Tensor

        # The number of points in image space (x,y) and kspace (fe,pe)
        nx = self.matrix_size
        ny = self.matrix_size
        nfe = self.oversampling * nx
        npe = ny

        # Go through all repetitions and create a trajectory and k-space
        kpe = []
        ktrue = []
        traj_kx = []
        traj_ky = []
        for _ in range(self.repetitions):
            if trajectory_type == 'cartesian':
                # Create Cartesian grid for k-space locations
                traj_ky_rep, traj_kx_rep, kpe_rep = self._cartesian_trajectory(npe, nfe, acceleration, sampling_order)
            elif trajectory_type == 'radial':
                # Create uniform radial trajectory
                traj_ky_rep, traj_kx_rep, kpe_rep = self._radial_trajectory(npe, nfe, acceleration)
            else:
                raise ValueError(f'Trajectory type {trajectory_type} not supported.')

            # Create analytic k-space and save trajectory
            ktrue.append(self.phantom.kspace(traj_ky_rep, traj_kx_rep))
            kpe.append(kpe_rep)
            traj_kx.append(traj_kx_rep)
            traj_ky.append(traj_ky_rep)

        # Reference image is the same for all repetitions
        im_dim = SpatialDimension(z=1, y=npe, x=nfe)
        self.imref = self.phantom.image_space(im_dim)

        # Multi-coil acquisition
        # TODO: proper application of coils
        ktrue = [repeat(k, '... -> coils ... ', coils=self.ncoils) for k in ktrue]

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
        ematrix.x = nfe
        ematrix.y = npe
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
        acq.resize(nfe, self.ncoils, trajectory_dimensions=2)
        acq.version = 1
        acq.available_channels = self.ncoils
        acq.center_sample = round(nfe / 2)
        acq.read_dir[0] = 1.0
        acq.phase_dir[1] = 1.0
        acq.slice_dir[2] = 1.0

        # Initialize an acquisition counter
        counter = 0

        # Write out a few noise scans
        for _ in range(32):
            noise = self.noise_level * torch.randn(self.ncoils, nfe, dtype=torch.complex64)
            # here's where we would make the noise correlated
            acq.scan_counter = counter
            acq.clearAllFlags()
            acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            acq.data[:] = noise.numpy()
            dset.append_acquisition(acq)
            counter += 1  # increment the scan counter

        # Loop over the repetitions, add noise and write to disk
        for rep in range(self.repetitions):
            noise = self.noise_level * torch.randn(self.ncoils, nfe, len(kpe[rep]), dtype=torch.complex64)
            # Here's where we would make the noise correlated
            K = ktrue[rep] + noise
            acq.idx.repetition = rep
            for pe_idx, pe_pos in enumerate(kpe[rep]):
                if not self.flag_invalid_reps or rep == 0 or pe_idx < len(kpe[rep]) // 2:  # fewer lines for rep > 0
                    # Set some fields in the header
                    acq.scan_counter = counter

                    # kpe is in the range [-npe//2, npe//2), the ismrmrd kspace_encoding_step_1 is in the range [0, npe)
                    kspace_encoding_step_1 = pe_pos + npe // 2
                    acq.idx.kspace_encode_step_1 = kspace_encoding_step_1
                    acq.clearAllFlags()
                    if kspace_encoding_step_1 == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif kspace_encoding_step_1 == npe - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)

                    # Set trajectory.
                    acq.traj[:] = (
                        torch.stack((traj_kx[rep][:, pe_idx], traj_ky[rep][:, pe_idx]), dim=1).numpy().astype(float)
                    )

                    # Set the data and append
                    acq.data[:] = K[:, :, pe_idx].numpy()
                    dset.append_acquisition(acq)
                    counter += 1

        # Clean up
        dset.close()

    @staticmethod
    def _cartesian_trajectory(
        npe: int,
        nfe: int,
        acceleration: int = 1,
        sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Cartesian sampling trajecgory.

        Parameters
        ----------
        npe
            number of phase encoding points before undersampling
        nfe
            number of frequency encoding points
        acceleration, optional
            undersampling factor, by default 1
        sampling_order, optional
            order how phase encoding points are sampled, by default "linear"
        """
        # Fully sampled frequency encoding
        kfe = torch.arange(-nfe // 2, nfe // 2)

        if sampling_order == 'random':
            # Linear order of a fully sampled kpe dimension. Undersampling is done later.
            kpe = torch.arange(0, npe)
        else:
            # Always include k-space center and more points on the negative side of k-space
            kpe_pos = torch.arange(0, npe // 2, acceleration)
            kpe_neg = -torch.arange(acceleration, npe // 2 + 1, acceleration)
            kpe = torch.cat((kpe_neg, kpe_pos), dim=0)

        # Different temporal orders of phase encoding points
        if sampling_order == 'random':
            perm = torch.randperm(len(kpe))
            kpe = kpe[perm[: len(perm) // acceleration]]
        elif sampling_order == 'linear':
            kpe, _ = torch.sort(kpe)
        elif sampling_order == 'low_high':
            idx = torch.argsort(torch.abs(kpe), stable=True)
            kpe = kpe[idx]
        elif sampling_order == 'high_low':
            idx = torch.argsort(-torch.abs(kpe), stable=True)
            kpe = kpe[idx]
        else:
            raise ValueError(f'sampling order {sampling_order} not supported.')

        # Combine frequency and phase encoding
        traj_ky, traj_kx = torch.meshgrid(kpe, kfe, indexing='xy')
        return traj_ky, traj_kx, kpe

    @staticmethod
    def _radial_trajectory(
        npe: int,
        nfe: int,
        acceleration: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate radial sampling trajecgory.

        Parameters
        ----------
        npe
            number of phase encoding points before undersampling, defines the number of angles
        nfe
            number of frequency encoding points, defines the sampling along each radial line
        acceleration, optional
            undersampling factor, by default 1
        """
        # Fully sampled frequency encoding
        kfe = torch.arange(-nfe // 2, nfe // 2)

        # Uniform angular sampling
        kpe = torch.linspace(0, npe - 1, npe // acceleration, dtype=torch.int32)
        kang = kpe * (torch.pi / len(kpe))

        traj_ky = torch.sin(kang[None, :]) * kfe[:, None]
        traj_kx = torch.cos(kang[None, :]) * kfe[:, None]
        return traj_ky, traj_kx, kpe
