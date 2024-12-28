import torch

from mrpro.data import SpatialDimension
from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.traj_calculators.KTrajectoryCalculator import KTrajectoryCalculator


class KTrajectorySpiral(KTrajectoryCalculator):
    """A Spiral variable density trajectory.

    Implements the spiral trajectory calculation as described in
    Simple Analytic Variable Density Spiral Design by Kim et al., MRM 2003"""

    def __init__(
        self,
        max_gradient: float,
        max_slewrate: float,
        fov: SpatialDimension | float,
        angle: float,
        acceleration_per_interleave: float = 1.0,
        density_factor: float = 1.0,
        gamma: float = 42577478,
    ):
        """Create a spiral trajectory calculator.

        Parameters
        ----------
        max_gradient
            Maximum gradient amplitude [T/m].
        max_slewrate
            Maximum slew rate [T/m/s].
        density_factor
            Density factor alpha.
        fov
            Field of view [m].
        acceleration_per_interleave
            Acceleration per interleave.
            Overall acceleration is (acceleration_per_interleave/n_interleaves),
            where n_interleaves is determined by k1_idx
        angle
            Angle between interleaves [rad].
            Usully set to 2pi/n_interleaves
        gamma
            Gyromagnetic ratio [Hz/T].
        """
        self.density_factor = density_factor
        self.max_gradient_gamma = max_gradient * gamma
        self.max_slewrate_gamma = max_slewrate * gamma
        self.acceleration_per_interleave = acceleration_per_interleave
        self.angle = angle

        if isinstance(fov, float):
            self.fov = fov
        elif fov.x != fov.y:
            raise ValueError('Only square FOV is supported.')
        elif fov.z != 0:
            raise ValueError('Only 2D trajectories are supported.')
        else:
            self.fov = fov.x

        if self.fov <= 0:
            raise ValueError('FOV must be positive.')
        if self.acceleration_per_interleave <= 0:
            raise ValueError('Acceleration per interleave must be positive.')
        if self.max_gradient_gamma <= 0:
            raise ValueError('Max gradient must be positive.')
        if self.max_slewrate_gamma <= 0:
            raise ValueError('Max slew rate must be positive.')
        if self.density_factor <= 0:
            raise ValueError('Density factor alpha must be positive.')

    def __call__(
        self,
        *,
        n_k0: int,
        k1_idx: torch.Tensor,
        encoding_matrix: SpatialDimension,
        **_,
    ) -> KTrajectory:
        """
        Calculate the spiral trajectory.

        Parameters
        ----------
        n_k0
            Number of samples along a spiral interleave.
        k1_idx
            Integer index of the interleaves
        encoding_matrix
            Dimensions of the encoding matrix.
            Only square matrices are supported.

        Returns
        -------
        Spiral Trajectory
        """
        if encoding_matrix.x != encoding_matrix.y:
            raise ValueError('Only square encoding matrices are supported.')
        if encoding_matrix.z != 1:
            raise ValueError('Only 2D trajectories are supported.')

        lam = 0.5 * (encoding_matrix.x / self.fov)
        n_turns = 1 / (
            1 - (1 - (2 * self.acceleration_per_interleave) / encoding_matrix.x) ** (1 / self.density_factor)
        )  # eq. 10
        max_angle = 2 * torch.pi * n_turns
        end_time_amplitude = (lam * max_angle) / (self.max_gradient_gamma * (self.density_factor + 1))  # eq. 5, Tes
        end_time_slew = torch.sqrt(lam * max_angle**2 / (self.max_slewrate_gamma)) / (
            self.density_factor / 2 + 1
        )  # eq. 8, Tea

        transition_time_slew_to_amplitude = (
            end_time_slew ** ((self.density_factor + 1) / (self.density_factor / 2 + 1))
            * (self.density_factor / 2 + 1)
            / end_time_amplitude
            / (self.density_factor + 1)
        ) ** (1 + 2 / self.density_factor)  # eq. 9, Ts2a

        has_amplitude_phase = transition_time_slew_to_amplitude < end_time_slew
        end_time = end_time_amplitude if has_amplitude_phase else end_time_slew

        def tau(t: torch.Tensor) -> torch.Tensor:
            """Normalized time function."""
            # eq. 11
            slew_phase = (t / end_time_slew) ** (1 / (self.density_factor / 2 + 1))
            slew_phase = slew_phase * ((t >= 0) * (t <= transition_time_slew_to_amplitude))
            if not has_amplitude_phase:
                return slew_phase
            amplitude_phase = (t / end_time_amplitude) ** (1 / (self.density_factor + 1))
            amplitude_phase = amplitude_phase * ((t > transition_time_slew_to_amplitude) * (t <= end_time_amplitude))
            return slew_phase + amplitude_phase

        t = torch.linspace(0, end_time, n_k0)
        tau_t = tau(t)
        k = lam * tau_t**self.density_factor * torch.exp(1j * max_angle * tau_t)  # eq. 2
        phase_rotation = torch.exp(self.angle * k1_idx)
        k = k[None, :] * phase_rotation[:, None]
        trajectory = KTrajectory(kx=k.real, ky=k.imag, kz=torch.zeros_like(k.real))
        return trajectory
