"""Create ismrmrd datasets."""

from pathlib import Path
from typing import Literal

import ismrmrd
import ismrmrd.xsd
import torch
from einops import repeat
from mrpro.data import SpatialDimension
from mrpro.phantoms import EllipsePhantom

from tests import RandomGenerator

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
        size of image matrix
    n_coils
        number of coils
    oversampling
        oversampling along readout (kx) direction
    repetitions
        number of repetitions,
    flag_invalid_reps
        flag to indicate that number of phase encoding steps are different for repetitions
    acceleration
        undersampling along phase encoding (ky)
    noise_level
        scaling factor for noise level
    trajectory_type
        cartesian
    sampling_order
        order how phase encoding points (ky) are obtained
    phantom
        phantom with different ellipses
    n_separate_calibration_lines
        number of additional calibration lines, linear Cartesian sampled
    """

    def __init__(
        self,
        filename: str | Path,
        matrix_size: int = 256,
        n_coils: int = 8,
        oversampling: int = 2,
        repetitions: int = 1,
        flag_invalid_reps: bool = False,
        acceleration: int = 1,
        noise_level: float = 0.00005,
        trajectory_type: Literal['cartesian', 'radial'] = 'cartesian',
        sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
        phantom: EllipsePhantom | None = None,
        add_bodycoil_acquisitions: bool = False,
        n_separate_calibration_lines: int = 0,
    ):
        if not phantom:
            phantom = EllipsePhantom()

        self.filename: str | Path = filename
        self.matrix_size: int = matrix_size
        self.n_coils: int = n_coils
        self.oversampling: int = oversampling
        self.repetitions: int = repetitions
        self.flag_invalid_reps: bool = flag_invalid_reps
        self.acceleration: int = acceleration
        self.noise_level: float = noise_level
        self.trajectory_type: Literal['cartesian', 'radial'] = trajectory_type
        self.sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = sampling_order
        self.phantom: EllipsePhantom = phantom
        self.img_ref: torch.Tensor
        self.n_separate_calibration_lines: int = n_separate_calibration_lines

        # The number of points in image space (x,y) and kspace (fe,pe)
        n_x = self.matrix_size
        n_y = self.matrix_size
        n_freq_encoding = self.oversampling * n_x
        n_phase_encoding = n_y

        # Go through all repetitions and create a trajectory and k-space
        kpe = []
        true_kspace = []
        traj_kx = []
        traj_ky = []
        for _ in range(self.repetitions):
            if trajectory_type == 'cartesian':
                # Create Cartesian grid for k-space locations
                traj_ky_rep, traj_kx_rep, kpe_rep = self._cartesian_trajectory(
                    n_phase_encoding,
                    n_freq_encoding,
                    acceleration,
                    sampling_order,
                )
            elif trajectory_type == 'radial':
                # Create uniform radial trajectory
                traj_ky_rep, traj_kx_rep, kpe_rep = self._radial_trajectory(
                    n_phase_encoding,
                    n_freq_encoding,
                    acceleration,
                )
            else:
                raise ValueError(f'Trajectory type {trajectory_type} not supported.')

            # Create analytic k-space and save trajectory
            true_kspace.append(self.phantom.kspace(traj_ky_rep, traj_kx_rep))
            kpe.append(kpe_rep)
            traj_kx.append(traj_kx_rep)
            traj_ky.append(traj_ky_rep)

        # Reference image is the same for all repetitions
        image_dimension = SpatialDimension(z=1, y=n_phase_encoding, x=n_freq_encoding)
        self.img_ref = self.phantom.image_space(image_dimension)

        # Multi-coil acquisition
        # TODO: proper application of coils
        true_kspace = [repeat(k, '... -> coils ... ', coils=self.n_coils) for k in true_kspace]

        # Open the dataset
        dataset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=True)

        # Create the XML header and write it to the file
        header = ismrmrd.xsd.ismrmrdHeader()

        # Experimental Conditions
        exp = ismrmrd.xsd.experimentalConditionsType()
        exp.H1resonanceFrequency_Hz = 128000000
        header.experimentalConditions = exp

        # Acquisition System Information
        sys = ismrmrd.xsd.acquisitionSystemInformationType()
        sys.receiverChannels = self.n_coils
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
        encoding_fov = ismrmrd.xsd.fieldOfViewMm()
        encoding_fov.x = self.oversampling * 256
        encoding_fov.y = 256
        encoding_fov.z = 5
        recon_fov = ismrmrd.xsd.fieldOfViewMm()
        recon_fov.x = 256
        recon_fov.y = 256
        recon_fov.z = 5

        encoding_matrix = ismrmrd.xsd.matrixSizeType()
        encoding_matrix.x = n_freq_encoding
        encoding_matrix.y = n_phase_encoding
        encoding_matrix.z = 1
        recon_matrix = ismrmrd.xsd.matrixSizeType()
        recon_matrix.x = n_x
        recon_matrix.y = n_y
        recon_matrix.z = 1

        encoding_space = ismrmrd.xsd.encodingSpaceType()
        encoding_space.matrixSize = encoding_matrix
        encoding_space.fieldOfView_mm = encoding_fov
        recon_space = ismrmrd.xsd.encodingSpaceType()
        recon_space.matrixSize = recon_matrix
        recon_space.fieldOfView_mm = recon_fov

        # Set encoded and recon spaces
        encoding.encodedSpace = encoding_space
        encoding.reconSpace = recon_space

        # Encoding limits
        limits = ismrmrd.xsd.encodingLimitsType()

        limits1 = ismrmrd.xsd.limitType()
        limits1.minimum = 0
        limits1.center = n_y // 2
        limits1.maximum = n_y - 1
        limits.kspace_encoding_step_1 = limits1

        limits_rep = ismrmrd.xsd.limitType()
        limits_rep.minimum = 0
        limits_rep.center = self.repetitions // 2
        limits_rep.maximum = self.repetitions - 1
        limits.repetition = limits_rep

        encoding.encodingLimits = limits
        header.encoding.append(encoding)

        dataset.write_xml_header(header.toXML('utf-8'))

        # Create an acquisition and reuse it
        acq = ismrmrd.Acquisition()
        acq.resize(n_freq_encoding, self.n_coils, trajectory_dimensions=2)
        acq.version = 1
        acq.available_channels = self.n_coils
        acq.center_sample = round(n_freq_encoding / 2)
        acq.read_dir[0] = 1.0
        acq.phase_dir[1] = 1.0
        acq.slice_dir[2] = 1.0

        scan_counter = 0

        # Write out a few noise scans
        for _ in range(32):
            noise = self.noise_level * torch.randn(self.n_coils, n_freq_encoding, dtype=torch.complex64)
            # here's where we would make the noise correlated
            acq.scan_counter = scan_counter
            acq.clearAllFlags()
            acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
            acq.data[:] = noise.numpy()
            dataset.append_acquisition(acq)
            scan_counter += 1

        # Add acquisitions obtained with a 2-element body coil (e.g. used for adjustment scans)
        if add_bodycoil_acquisitions:
            acq.resize(n_freq_encoding, 2, trajectory_dimensions=2)
            for _ in range(8):
                acq.scan_counter = scan_counter
                acq.clearAllFlags()
                acq.data[:] = torch.randn(2, n_freq_encoding, dtype=torch.complex64)
                dataset.append_acquisition(acq)
                scan_counter += 1
            acq.resize(n_freq_encoding, self.n_coils, trajectory_dimensions=2)

        # Calibration lines
        if n_separate_calibration_lines > 0:
            traj_ky_calibration, traj_kx_calibration, kpe_calibration = self._cartesian_trajectory(
                n_separate_calibration_lines,
                n_freq_encoding,
                1,
                'linear',
            )
            kspace_calibration = self.phantom.kspace(traj_ky_calibration, traj_kx_calibration)
            kspace_calibration = repeat(kspace_calibration, '... -> coils ... ', coils=self.n_coils)
            kspace_calibration = kspace_calibration + self.noise_level * torch.randn(
                self.n_coils, n_freq_encoding, len(kpe_calibration), dtype=torch.complex64
            )

            for pe_idx, pe_pos in enumerate(kpe_calibration):
                # Set some fields in the header
                acq.scan_counter = scan_counter

                # kpe is in the range [-npe//2, npe//2), the ismrmrd kspace_encoding_step_1 is in the range [0, npe)
                kspace_encoding_step_1 = pe_pos + n_phase_encoding // 2
                acq.idx.kspace_encode_step_1 = kspace_encoding_step_1
                acq.clearAllFlags()
                acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)

                # Set the data and append
                acq.data[:] = kspace_calibration[:, :, pe_idx].numpy()
                dataset.append_acquisition(acq)
                scan_counter += 1

        # Loop over the repetitions, add noise and write to disk
        for rep in range(self.repetitions):
            noise = self.noise_level * torch.randn(self.n_coils, n_freq_encoding, len(kpe[rep]), dtype=torch.complex64)
            # Here's where we would make the noise correlated
            kspace_with_noise = true_kspace[rep] + noise
            acq.idx.repetition = rep
            for pe_idx, pe_pos in enumerate(kpe[rep]):
                if not self.flag_invalid_reps or rep == 0 or pe_idx < len(kpe[rep]) // 2:  # fewer lines for rep > 0
                    # Set some fields in the header
                    acq.scan_counter = scan_counter

                    # kpe is in the range [-npe//2, npe//2), the ismrmrd kspace_encoding_step_1 is in the range [0, npe)
                    kspace_encoding_step_1 = pe_pos + n_phase_encoding // 2
                    acq.idx.kspace_encode_step_1 = kspace_encoding_step_1
                    acq.clearAllFlags()
                    if kspace_encoding_step_1 == 0:
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_ENCODE_STEP1)
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_SLICE)
                        acq.setFlag(ismrmrd.ACQ_FIRST_IN_REPETITION)
                    elif kspace_encoding_step_1 == n_phase_encoding - 1:
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_ENCODE_STEP1)
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
                        acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)

                    # Set trajectory.
                    acq.traj[:] = (
                        torch.stack((traj_kx[rep][:, pe_idx], traj_ky[rep][:, pe_idx]), dim=1).numpy().astype(float)
                    )

                    # Set the data and append
                    acq.data[:] = kspace_with_noise[:, :, pe_idx].numpy()
                    dataset.append_acquisition(acq)
                    scan_counter += 1

        # Clean up
        dataset.close()

    @staticmethod
    def _cartesian_trajectory(
        n_phase_encoding: int,
        n_freq_encoding: int,
        acceleration: int = 1,
        sampling_order: Literal['linear', 'low_high', 'high_low', 'random'] = 'linear',
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate Cartesian sampling trajectory.

        Parameters
        ----------
        n_phase_encoding
            number of phase encoding points before undersampling
        n_freq_encoding
            number of frequency encoding points
        acceleration
            undersampling factor
        sampling_order
            order how phase encoding points are sampled
        """
        # Fully sampled frequency encoding
        kfe = torch.arange(-n_freq_encoding // 2, n_freq_encoding // 2)

        if sampling_order == 'random':
            # Linear order of a fully sampled kpe dimension. Undersampling is done later.
            kpe = torch.arange(0, n_phase_encoding)
        else:
            # Always include k-space center and more points on the negative side of k-space
            kpe_pos = torch.arange(0, n_phase_encoding // 2, acceleration)
            kpe_neg = -torch.arange(acceleration, n_phase_encoding // 2 + 1, acceleration)
            kpe = torch.cat((kpe_neg, kpe_pos), dim=0)

        # Different temporal orders of phase encoding points
        if sampling_order == 'random':
            perm = RandomGenerator(13).randperm(len(kpe))
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
        n_phase_encoding: int,
        n_freq_encoding: int,
        acceleration: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate radial sampling trajectory.

        Parameters
        ----------
        n_phase_encoding
            number of phase encoding points before undersampling, defines the number of angles
        n_freq_encoding
            number of frequency encoding points, defines the sampling along each radial line
        acceleration
            undersampling factor
        """
        # Fully sampled frequency encoding (sorting of ISMRMD is x,y,z)
        kfe = repeat(torch.arange(-n_freq_encoding // 2, n_freq_encoding // 2), 'k0 -> k0 k1', k1=1)

        # Uniform angular sampling (sorting of ISMRMD is x,y,z)
        kpe = torch.linspace(0, n_phase_encoding - 1, n_phase_encoding // acceleration, dtype=torch.int32)
        kang = repeat(kpe * (torch.pi / len(kpe)), 'k1 -> k0 k1', k0=1)

        traj_ky = torch.sin(kang) * kfe
        traj_kx = torch.cos(kang) * kfe
        return traj_ky, traj_kx, kpe
