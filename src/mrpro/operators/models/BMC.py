"""Bloch-McConnell simulation."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from mrpro.data.Dataclass import Dataclass
from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.reshape import unsqueeze_left, unsqueeze_right
from mrpro.utils.slice_profiles import SliceProfileBase
from mrpro.utils.TensorAttributeMixin import TensorAttributeMixin
from mrpro.utils.unit_conversion import GYROMAGNETIC_RATIO_PROTON


@dataclass
class MTSaturation(ABC, TensorAttributeMixin):
    """Base class for MT lineshape models."""

    pool_index: int
    """Index of the MT pool in the pool dimension."""

    t2: torch.Tensor
    """Transverse relaxation time in seconds."""

    @abstractmethod
    def __call__(self, delta_omega: torch.Tensor) -> torch.Tensor:
        r"""Evaluate \(G(\Delta)\) [s]."""


@dataclass
class LorentzianMT(MTSaturation):
    """Lorentzian lineshape for MT saturation."""

    def __call__(self, delta_omega: torch.Tensor) -> torch.Tensor:
        r"""Evaluate \(G(\Delta)\) [s].

        Parameters
        ----------
        delta_omega
            Detuning in rad/s.

        Returns
        -------
        g
            Lineshape value in seconds.
        """
        t2 = self.t2.to(delta_omega)
        x = delta_omega * t2
        return t2 / (1 + x * x)


@dataclass
class SuperLorentzianMT(MTSaturation):
    """Super-Lorentzian lineshape for MT saturation."""

    samples: int = 101
    """Quadrature samples for numerical integration."""

    def __call__(self, delta_omega: torch.Tensor) -> torch.Tensor:
        r"""Evaluate \(G(\Delta)\) [s].

        Parameters
        ----------
        delta_omega
            Detuning in rad/s.
        t2
            Transverse relaxation time in seconds.
        """
        t2 = self.t2.to(delta_omega)
        u = torch.linspace(0.0, 1.0, self.samples, device=delta_omega.device, dtype=delta_omega.dtype)
        du = u[1] - u[0]
        denom = (3 * u * u - 1).abs().clamp_min(1e-12)
        x = (delta_omega[..., None] * t2[..., None]) / denom
        integrand = (2.0 / torch.pi) ** 0.5 * t2[..., None] / denom * torch.exp(-2 * x * x)
        return integrand.sum(dim=-1) * torch.pi * du


class Parameters(Dataclass):
    """Parameters for Bloch-McConnell simulation.

    Shapes
    ------
    - poolwise: ``(..., pools)``

    Notes
    -----
    Hyperpolarization is handled by setting a non-equilibrium initial state
    (via ``initial_state(..., mz=...)`` or ``ResetBlock(state=...)``) and by
    choosing ``equilibrium_magnetization`` appropriately.
    """

    equilibrium_magnetization: torch.Tensor
    """Equilibrium magnetization."""
    t1: torch.Tensor
    """T1 relaxation time in seconds.
    Shape ``(..., pools)``.
    """
    t2: torch.Tensor
    """T2 relaxation time in seconds. Shape ``(..., pools)``."""

    exchange_rate: torch.Tensor
    """Exchange rate in 1/s.
    Shape ``(..., pools, pools)``
    where element ``[..., i, j]`` is the ratefrom pool j to pool i."""

    chemical_shift: torch.Tensor | None = None
    """Chemical shift in Hz. Shape ``(..., pools)``."""

    static_off_resonance: torch.Tensor | None = None
    """Delta B0 in rad/s. Shape ``(...)`` (global per voxel/batch)."""

    relative_b1: torch.Tensor | None = None
    """Relative B1 scaling factor. Shape ``(...)`` (global per voxel/batch)."""

    mt_saturation: MTSaturation | None = None
    """MT saturation model. Shape ``(..., pools)``. """

    @property
    def n_pools(self) -> int:
        """Number of pools."""
        return int(self.equilibrium_magnetization.shape[-1])

    @property
    def ndim(self) -> int:
        """Broadcast ndim of parameter batch dimensions."""
        ndim = max(
            self.equilibrium_magnetization.ndim,
            self.t1.ndim,
            self.t2.ndim,
            self.exchange_rate.ndim - 1,
        )
        if self.chemical_shift is not None:
            ndim = max(ndim, self.chemical_shift.ndim)
        if self.static_off_resonance is not None:
            ndim = max(ndim, self.static_off_resonance.ndim + 1)
        if self.relative_b1 is not None:
            ndim = max(ndim, self.relative_b1.ndim + 1)
        return ndim


def system_recovery_vector(parameters: Parameters) -> torch.Tensor:
    """Build the affine recovery vector."""
    m0, t1 = parameters.equilibrium_magnetization, parameters.t1
    batch_shape = torch.broadcast_shapes(
        m0.shape[:-1],
        t1.shape[:-1],
        parameters.t2.shape[:-1],
        parameters.exchange_rate.shape[:-2],
    )
    if parameters.chemical_shift is not None:
        batch_shape = torch.broadcast_shapes(batch_shape, parameters.chemical_shift.shape[:-1])
    if parameters.static_off_resonance is not None:
        batch_shape = torch.broadcast_shapes(batch_shape, parameters.static_off_resonance.shape)
    if parameters.relative_b1 is not None:
        batch_shape = torch.broadcast_shapes(batch_shape, parameters.relative_b1.shape)
    c = torch.zeros(*batch_shape, 3 * parameters.n_pools, device=m0.device, dtype=m0.dtype)
    c[..., 2 * parameters.n_pools :] = (1.0 / t1) * m0
    return c


def initial_state(parameters: Parameters, mz: torch.Tensor | None = None) -> torch.Tensor:
    """Create an initial magnetization state.

    Parameters
    ----------
    parameters
        Simulation parameters.
    mz
        Optional initial longitudinal magnetization, shape ``(..., pools)``.
        If omitted, uses equilibrium_magnetization.

    Returns
    -------
    state
        Tensor with shape ``(..., isochromats=1, pools, 3)`` holding ``(Mx, My, Mz)``.
    """
    mz0 = parameters.equilibrium_magnetization if mz is None else mz
    mz0 = mz0.to(parameters.equilibrium_magnetization)
    z = torch.zeros_like(mz0)
    return torch.stack([z, z, mz0], dim=-1).unsqueeze(-3)


def exchange_generator(exchange_rate: torch.Tensor) -> torch.Tensor:
    r"""Construct exchange generator \(Q\) for \(dM/dt = Q M\).

    Parameters
    ----------
    exchange_rate
        Shape ``(..., pools, pools)`` with element ``[..., i, j]`` the rate
        from pool j to pool i.

    Returns
    -------
    q
        Shape ``(..., pools, pools)``.
    """
    exchange_rate = torch.as_tensor(exchange_rate)
    out_rate = exchange_rate.sum(dim=-2)
    return exchange_rate - torch.diag_embed(out_rate)


def system_base_matrix(
    parameters: Parameters,
    rf_frequency: torch.Tensor | float,
    extra_off_resonance: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Build the RF-amplitude independent Bloch-McConnell matrix."""
    m0, t1, t2, exchange = parameters.equilibrium_magnetization, parameters.t1, parameters.t2, parameters.exchange_rate
    freq = torch.as_tensor(rf_frequency, device=m0.device, dtype=m0.dtype)
    extra_dw = torch.as_tensor(extra_off_resonance, device=m0.device, dtype=m0.dtype)

    if parameters.chemical_shift is not None:
        shift = parameters.chemical_shift.to(m0)
    else:
        shift = m0.new_zeros(*m0.shape[:-1], parameters.n_pools)

    if parameters.static_off_resonance is not None:
        dw0 = parameters.static_off_resonance.to(m0)
    else:
        dw0 = m0.new_zeros(m0.shape[:-1])

    batch = torch.broadcast_shapes(
        m0.shape[:-1],
        t1.shape[:-1],
        t2.shape[:-1],
        exchange.shape[:-2],
        freq.shape,
        extra_dw.shape,
        shift.shape[:-1],
        dw0.shape,
    )
    t1 = torch.broadcast_to(t1, (*batch, parameters.n_pools))
    t2 = torch.broadcast_to(t2, (*batch, parameters.n_pools))
    exchange = torch.broadcast_to(exchange, (*batch, parameters.n_pools, parameters.n_pools))
    shift = torch.broadcast_to(shift, (*batch, parameters.n_pools))
    dw0 = torch.broadcast_to(dw0, batch)
    freq = torch.broadcast_to(freq, batch)
    extra_dw = torch.broadcast_to(extra_dw, batch)

    r1 = 1.0 / t1
    r2 = 1.0 / t2

    qz = exchange_generator(exchange)
    qxy = qz
    if parameters.mt_saturation is not None:
        if not (0 <= parameters.mt_saturation.pool_index < parameters.n_pools):
            raise ValueError('mt_saturation.pool_index out of bounds.')
        qxy = qz.clone()
        qxy[..., parameters.mt_saturation.pool_index, :] = 0
        qxy[..., :, parameters.mt_saturation.pool_index] = 0

    delta_omega = dw0[..., None] + extra_dw[..., None] - 2 * torch.pi * freq[..., None] + 2 * torch.pi * shift

    a_xx = qxy - torch.diag_embed(r2)
    a_zz = qz - torch.diag_embed(r1)
    a_xy = -torch.diag_embed(delta_omega)

    n = 3 * parameters.n_pools
    matrix = torch.zeros(*batch, n, n, device=m0.device, dtype=m0.dtype)
    matrix[..., : parameters.n_pools, : parameters.n_pools] = a_xx
    matrix[..., parameters.n_pools : 2 * parameters.n_pools, parameters.n_pools : 2 * parameters.n_pools] = a_xx
    matrix[..., 2 * parameters.n_pools :, 2 * parameters.n_pools :] = a_zz
    matrix[..., : parameters.n_pools, parameters.n_pools : 2 * parameters.n_pools] += a_xy
    matrix[..., parameters.n_pools : 2 * parameters.n_pools, : parameters.n_pools] -= a_xy
    return matrix


def gradient_to_extra_off_resonance(
    gradient_z: torch.Tensor | float | None,
    gradient_y: torch.Tensor | float | None,
    gradient_x: torch.Tensor | float | None,
    positions: SpatialDimension[torch.Tensor],
) -> torch.Tensor:
    """Convert gradients and isochromat positions to extra off-resonance in rad/s."""
    device = positions.z.device
    gradient_z = torch.as_tensor(0.0 if gradient_z is None else gradient_z, device=device)
    gradient_y = torch.as_tensor(0.0 if gradient_y is None else gradient_y, device=device)
    gradient_x = torch.as_tensor(0.0 if gradient_x is None else gradient_x, device=device)
    px = positions.x.to(device)
    py = positions.y.to(device)
    pz = positions.z.to(device)
    return (
        2
        * torch.pi
        * GYROMAGNETIC_RATIO_PROTON
        * (gradient_z[..., None] * pz + gradient_y[..., None] * py + gradient_x[..., None] * px)
    )


def system_rf_matrix(
    parameters: Parameters,
    rf_amplitude: torch.Tensor | float,
    rf_phase: torch.Tensor | float,
    rf_frequency: torch.Tensor | float,
    extra_off_resonance: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Build the RF-amplitude dependent Bloch-McConnell matrix contribution."""
    m0 = parameters.equilibrium_magnetization

    amp = torch.as_tensor(rf_amplitude, device=m0.device, dtype=m0.dtype)
    phase = torch.as_tensor(rf_phase, device=m0.device, dtype=m0.dtype)
    freq = torch.as_tensor(rf_frequency, device=m0.device, dtype=m0.dtype)
    extra_dw = torch.as_tensor(extra_off_resonance, device=m0.device, dtype=m0.dtype)

    if parameters.relative_b1 is not None:
        rb1 = parameters.relative_b1.to(amp)
        if rb1.is_complex():
            phase = phase + rb1.angle()
            amp = amp * rb1.abs()
        else:
            amp = amp * rb1

    if parameters.chemical_shift is not None:
        shift = parameters.chemical_shift.to(m0)
    else:
        shift = m0.new_zeros(*m0.shape[:-1], parameters.n_pools)

    if parameters.static_off_resonance is not None:
        dw0 = parameters.static_off_resonance.to(m0)
    else:
        dw0 = m0.new_zeros(m0.shape[:-1])

    batch = torch.broadcast_shapes(
        m0.shape[:-1],
        amp.shape,
        phase.shape,
        freq.shape,
        extra_dw.shape,
        shift.shape[:-1],
        dw0.shape,
    )
    shift = torch.broadcast_to(shift, (*batch, parameters.n_pools))
    dw0 = torch.broadcast_to(dw0, batch)
    amp = torch.broadcast_to(amp, batch)
    phase = torch.broadcast_to(phase, batch)
    freq = torch.broadcast_to(freq, batch)
    extra_dw = torch.broadcast_to(extra_dw, batch)

    delta_omega = dw0[..., None] + extra_dw[..., None] - 2 * torch.pi * freq[..., None] + 2 * torch.pi * shift

    w1 = 2 * torch.pi * amp
    w1x = w1 * torch.cos(phase)
    w1y = w1 * torch.sin(phase)

    eye_rf = torch.eye(parameters.n_pools, device=m0.device, dtype=m0.dtype)
    if parameters.mt_saturation is not None:
        eye_rf = eye_rf.clone()
        eye_rf[parameters.mt_saturation.pool_index, parameters.mt_saturation.pool_index] = 0.0

    a_xz = -w1y[..., None, None] * eye_rf
    a_yz = w1x[..., None, None] * eye_rf

    n = 3 * parameters.n_pools
    matrix = torch.zeros(*batch, n, n, device=m0.device, dtype=m0.dtype)
    matrix[..., : parameters.n_pools, 2 * parameters.n_pools :] += a_xz
    matrix[..., parameters.n_pools : 2 * parameters.n_pools, 2 * parameters.n_pools :] += a_yz
    matrix[..., 2 * parameters.n_pools :, : parameters.n_pools] -= a_xz
    matrix[..., 2 * parameters.n_pools :, parameters.n_pools : 2 * parameters.n_pools] -= a_yz

    if parameters.mt_saturation is not None:
        g = parameters.mt_saturation(delta_omega[..., parameters.mt_saturation.pool_index])
        one_hot = torch.zeros(parameters.n_pools, device=m0.device, dtype=m0.dtype)
        one_hot[parameters.mt_saturation.pool_index] = 1.0
        mt_diag = torch.diag_embed(((w1 * w1) * g)[..., None] * one_hot)
        matrix[..., 2 * parameters.n_pools :, 2 * parameters.n_pools :] -= mt_diag

    return matrix


def system_matrix(
    parameters: Parameters,
    rf_amplitude: torch.Tensor | float,
    rf_phase: torch.Tensor | float,
    rf_frequency: torch.Tensor | float,
    extra_off_resonance: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Build affine Bloch-McConnell system \(dm/dt = A m + c\).

    Parameters
    ----------
    parameters
        Simulation parameters.
    rf_amplitude
        RF amplitude in Hz, broadcastable to batch. Shape ``(...)``.
    rf_phase
        RF phase in rad, broadcastable to batch. Shape ``(...)``.
    rf_frequency
        RF carrier offset in Hz, broadcastable to batch. Shape ``(...)``.
    extra_off_resonance
        Additional off-resonance in rad/s, broadcastable to batch. Shape ``(...)``.

    Returns
    -------
    A
        System matrix with shape ``(..., 3*pools, 3*pools)``.
    c
        Inhomogeneity vector with shape ``(..., 3*pools)``.
    """
    matrix = system_base_matrix(parameters, rf_frequency, extra_off_resonance) + system_rf_matrix(
        parameters, rf_amplitude, rf_phase, rf_frequency, extra_off_resonance
    )
    c = system_recovery_vector(parameters)
    return matrix, c


def propagate(
    state: torch.Tensor,
    matrix: torch.Tensor,
    c: torch.Tensor,
    duration: torch.Tensor | float,
) -> torch.Tensor:
    r"""Propagate dynamics \(dm/dt = A m + c\) via exact affine evolution."""
    matrix = matrix.unsqueeze(-3)
    c = c.unsqueeze(-2)
    duration = torch.as_tensor(duration, device=matrix.device, dtype=matrix.dtype)
    step = propagation_step(matrix, c, duration)
    return apply_propagation_step(state, step)


def propagation_step(
    matrix: torch.Tensor,
    c: torch.Tensor,
    duration: torch.Tensor | float,
) -> torch.Tensor:
    """Build exact affine propagation steps for constant-system evolution."""
    duration = torch.as_tensor(duration, device=matrix.device, dtype=matrix.dtype)
    duration = unsqueeze_right(duration, matrix.ndim - duration.ndim)
    linear_step = torch.matrix_exp(matrix * duration)
    identity = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    offset_rhs = ((linear_step - identity) @ c.unsqueeze(-1)).squeeze(-1)
    offset, info = torch.linalg.solve_ex(matrix, offset_rhs.unsqueeze(-1))
    if torch.any(info != 0):
        augmented = torch.zeros(
            *matrix.shape[:-2],
            matrix.shape[-1] + 1,
            matrix.shape[-1] + 1,
            device=matrix.device,
            dtype=matrix.dtype,
        )
        augmented[..., :-1, :-1] = matrix
        augmented[..., :-1, -1] = c
        augmented_step = torch.matrix_exp(augmented * duration)
        return augmented_step[..., :-1, :]
    return torch.cat([linear_step, offset], dim=-1)


def apply_propagation_step(state: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    """Apply a precomputed exact affine propagation step to a state."""
    pools = int(state.shape[-2])
    isochromats = int(state.shape[-3])
    n = 3 * pools
    if state.shape[:-3] == step.shape[:-3]:
        batch = state.shape[:-3]
    else:
        batch = torch.broadcast_shapes(state.shape[:-3], step.shape[:-3])
        state = torch.broadcast_to(state, (*batch, isochromats, pools, 3))
        step = torch.broadcast_to(step, (*batch, isochromats, n, n + 1))
    m = state.mT.reshape(*batch, isochromats, n)
    linear_step, offset = step[..., :n], step[..., n]
    m_next = (linear_step @ m.unsqueeze(-1)).squeeze(-1) + offset
    return m_next.reshape(*batch, isochromats, 3, pools).mT


def transverse_readout(state: torch.Tensor) -> torch.Tensor:
    """Complex transverse readout per pool, averaged over isochromats."""
    return torch.complex(state[..., 0], state[..., 1]).mean(dim=-2)


class BMCBlock(TensorAttributeMixin, ABC):
    """Base class for Bloch-McConnell blocks."""

    def __call__(
        self,
        parameters: Parameters,
        state: torch.Tensor | None = None,
        *,
        zero_matrix: torch.Tensor | None = None,
        zero_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        if state is None:
            state = initial_state(parameters)
        if zero_matrix is None and zero_c is None:
            return super().__call__(parameters, state)
        return super().__call__(parameters, state, zero_matrix=zero_matrix, zero_c=zero_c)

    @abstractmethod
    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        raise NotImplementedError

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.as_tensor(0.0)


class ConstantRFBlock(BMCBlock):
    """Constant RF block for a duration."""

    def __init__(
        self,
        duration: torch.Tensor | float,
        rf_amplitude: torch.Tensor | float,
        rf_phase: torch.Tensor | float = 0.0,
        rf_frequency: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize the block.

        Parameters
        ----------
        duration
            Duration in seconds.
        rf_amplitude
            RF amplitude in Hz.
        rf_phase
            RF phase in rad.
        rf_frequency
            RF frequency in Hz.
        """
        super().__init__()
        self._duration = torch.as_tensor(duration)
        self.rf_amplitude = torch.as_tensor(rf_amplitude)
        self.rf_phase = torch.as_tensor(rf_phase)
        self.rf_frequency = torch.as_tensor(rf_frequency)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self._duration

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        matrix, c = system_matrix(
            parameters,
            self.rf_amplitude.to(state),
            self.rf_phase.to(state),
            self.rf_frequency.to(state),
        )
        state = propagate(state, matrix, c, self._duration.to(state))
        return state, ()


class PiecewiseRFBlock(BMCBlock):
    """Piecewise-constant RF block."""

    def __init__(
        self,
        rf_amplitude: torch.Tensor,
        rf_phase: torch.Tensor | float = 0.0,
        rf_frequency: torch.Tensor | float = 0.0,
        dt: torch.Tensor | float = 0.0,
        extra_off_resonance: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize the block.

        Parameters
        ----------
        rf_amplitude
            RF amplitude. Shape ``(time, ...)``.
        rf_phase
            RF phase in rad. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        rf_frequency
            RF frequency in Hz. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        dt
            Sample duration in seconds. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        extra_off_resonance
            Additional off-resonance in rad/s. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        """
        super().__init__()
        self.rf_amplitude = torch.as_tensor(rf_amplitude)
        self.rf_phase = torch.as_tensor(rf_phase)
        self.rf_frequency = torch.as_tensor(rf_frequency)
        self.dt = torch.as_tensor(dt)
        self.extra_off_resonance = torch.as_tensor(extra_off_resonance)

        if self.rf_amplitude.ndim < 1:
            raise ValueError('rf_amplitude must have a leading time dimension.')

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        if self.dt.ndim == 0:
            return self.dt * self.rf_amplitude.shape[0]
        if self.dt.shape[0] == 1:
            return self.dt.squeeze(0) * self.rf_amplitude.shape[0]
        return self.dt.sum(dim=0)

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        rf_amplitude = self.rf_amplitude
        rf_phase = unsqueeze_left(self.rf_phase, max(0, 1 - self.rf_phase.ndim))
        rf_frequency = unsqueeze_left(self.rf_frequency, max(0, 1 - self.rf_frequency.ndim))
        dt = unsqueeze_left(self.dt, max(0, 1 - self.dt.ndim))
        extra_off_resonance = unsqueeze_left(self.extra_off_resonance, max(0, 1 - self.extra_off_resonance.ndim))

        time = rf_amplitude.shape[0]
        for name, tensor in (
            ('rf_phase', rf_phase),
            ('rf_frequency', rf_frequency),
            ('dt', dt),
            ('extra_off_resonance', extra_off_resonance),
        ):
            if tensor.shape[0] not in (1, time):
                raise ValueError(f'{name} must have leading dimension 1 or match rf_amplitude.')

        tensor_ndim = max(
            rf_amplitude.ndim,
            rf_phase.ndim,
            rf_frequency.ndim,
            dt.ndim,
            extra_off_resonance.ndim,
            state.ndim - 2,
            parameters.ndim,
        )

        rf_amplitude = unsqueeze_right(rf_amplitude, tensor_ndim - rf_amplitude.ndim)
        rf_phase = unsqueeze_right(rf_phase, tensor_ndim - rf_phase.ndim)
        rf_frequency = unsqueeze_right(rf_frequency, tensor_ndim - rf_frequency.ndim)
        dt = unsqueeze_right(dt, tensor_ndim - dt.ndim)
        extra_off_resonance = unsqueeze_right(extra_off_resonance, tensor_ndim - extra_off_resonance.ndim)

        if time != 1:
            if rf_phase.shape[0] == 1:
                rf_phase = rf_phase.expand(time, *rf_phase.shape[1:])
            if rf_frequency.shape[0] == 1:
                rf_frequency = rf_frequency.expand(time, *rf_frequency.shape[1:])
            if dt.shape[0] == 1:
                dt = dt.expand(time, *dt.shape[1:])
            if extra_off_resonance.shape[0] == 1:
                extra_off_resonance = extra_off_resonance.expand(time, *extra_off_resonance.shape[1:])

        c = system_recovery_vector(parameters)
        same_frequency = bool(torch.all(rf_frequency == rf_frequency[:1]))
        same_extra = bool(torch.all(extra_off_resonance == extra_off_resonance[:1]))
        base_matrix = (
            system_base_matrix(parameters, rf_frequency[0], extra_off_resonance[0])
            if same_frequency and same_extra
            else None
        )

        batch_shape = torch.broadcast_shapes(
            rf_amplitude.shape[1:],
            rf_phase.shape[1:],
            rf_frequency.shape[1:],
            dt.shape[1:],
            extra_off_resonance.shape[1:],
        )
        batch_size = 1
        for size in batch_shape:
            batch_size *= size
        work = time * batch_size * (3 * parameters.n_pools) ** 2
        chunk_size = time if work <= 2_500_000 else (128 if same_frequency else 64)

        def run_chunk(
            state: torch.Tensor,
            rf_amplitude: torch.Tensor,
            rf_phase: torch.Tensor,
            rf_frequency: torch.Tensor,
            dt_chunk: torch.Tensor,
            extra_off_resonance: torch.Tensor,
        ) -> torch.Tensor:
            if same_frequency:
                if base_matrix is not None:
                    matrices = base_matrix + system_rf_matrix(
                        parameters,
                        rf_amplitude,
                        rf_phase,
                        rf_frequency,
                        extra_off_resonance,
                    )
                else:
                    matrices = system_base_matrix(
                        parameters,
                        rf_frequency,
                        extra_off_resonance,
                    ) + system_rf_matrix(
                        parameters,
                        rf_amplitude,
                        rf_phase,
                        rf_frequency,
                        extra_off_resonance,
                    )
            else:
                matrices = system_base_matrix(
                    parameters,
                    rf_frequency,
                    extra_off_resonance,
                ) + system_rf_matrix(
                    parameters,
                    rf_amplitude,
                    rf_phase,
                    rf_frequency,
                    extra_off_resonance,
                )
            matrices = matrices.unsqueeze(-3)
            recovery = c[(None,) * (matrices.ndim - c.ndim - 2) + (...,)].unsqueeze(-2)
            recovery = torch.broadcast_to(recovery, (*matrices.shape[:-2], c.shape[-1]))
            steps = propagation_step(matrices, recovery, dt_chunk)
            for step in steps:
                state = apply_propagation_step(state, step)
            return state

        n_chunks = (time + chunk_size - 1) // chunk_size
        for rf_amplitude_chunk, rf_phase_chunk, rf_frequency_chunk, dt_chunk, extra_off_resonance_chunk in zip(
            rf_amplitude.tensor_split(n_chunks),
            rf_phase.tensor_split(n_chunks),
            rf_frequency.tensor_split(n_chunks),
            dt.tensor_split(n_chunks),
            extra_off_resonance.tensor_split(n_chunks),
            strict=True,
        ):
            state = activation_checkpoint(
                run_chunk,
                state,
                rf_amplitude_chunk,
                rf_phase_chunk,
                rf_frequency_chunk,
                dt_chunk,
                extra_off_resonance_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        return state, ()


class SliceSelectiveRFBlock(BMCBlock):
    """Piecewise RF block with an effective image-space slice profile."""

    def __init__(
        self,
        rf_amplitude: torch.Tensor,
        slice_profile: SliceProfileBase,
        positions: SpatialDimension[torch.Tensor],
        rf_phase: torch.Tensor | float = 0.0,
        rf_frequency: torch.Tensor | float = 0.0,
        dt: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize the block.

        Parameters
        ----------
        rf_amplitude
            RF amplitude. Shape ``(time, ...)``.
        slice_profile
            Slice profile evaluated at ``positions.z`` to scale excitation across isochromats.
        positions
            Isochromat positions in meters for ``(z, y, x)``.
        rf_phase
            RF phase in rad. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        rf_frequency
            RF frequency in Hz. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        dt
            Sample duration in seconds. Shape ``(time, ...)``, ``(1, ...)`` or scalar.
        """
        super().__init__()
        self._block = PiecewiseRFBlock(rf_amplitude=rf_amplitude, rf_phase=rf_phase, rf_frequency=rf_frequency, dt=dt)
        self.slice_profile = slice_profile
        self.positions = positions.apply(torch.as_tensor)
        self.n_iso = max(int(self.positions.z.numel()), int(self.positions.y.numel()), int(self.positions.x.numel()))
        if any(axis.numel() not in (1, self.n_iso) for axis in (self.positions.z, self.positions.y, self.positions.x)):
            raise ValueError(
                'positions.x, positions.y and positions.z must each have length 1 or match the isochromat count.'
            )

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self._block.duration

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        n_iso = state.shape[-3]
        if n_iso != self.n_iso:
            raise ValueError(
                f'Isochromat axis mismatch: state has {n_iso} isochromats but positions define {self.n_iso}.'
            )

        profile = self.slice_profile(self.positions.z.to(state))
        effective_rf_amplitude = self._block.rf_amplitude[..., None] * profile
        if parameters.relative_b1 is not None:
            effective_rf_amplitude = effective_rf_amplitude * parameters.relative_b1[..., None]
        effective_block = PiecewiseRFBlock(
            rf_amplitude=effective_rf_amplitude,
            rf_phase=self._block.rf_phase,
            rf_frequency=self._block.rf_frequency,
            dt=self._block.dt,
            extra_off_resonance=self._block.extra_off_resonance,
        )
        effective_parameters = Parameters(
            equilibrium_magnetization=parameters.equilibrium_magnetization,
            t1=parameters.t1,
            t2=parameters.t2,
            exchange_rate=parameters.exchange_rate,
            chemical_shift=parameters.chemical_shift,
            static_off_resonance=parameters.static_off_resonance,
            relative_b1=None,
            mt_saturation=parameters.mt_saturation,
        )
        state, outputs = effective_block(effective_parameters, state.unsqueeze(-3))
        return state.squeeze(-3), outputs


class DelayBlock(BMCBlock):
    """Delay without RF."""

    def __init__(self, duration: torch.Tensor | float) -> None:
        """Initialize the block.

        Parameters
        ----------
        duration
            Duration in seconds. Shape ``(..., pools)``.
        """
        super().__init__()
        self._duration = torch.as_tensor(duration)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self._duration

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        *,
        zero_matrix: torch.Tensor | None = None,
        zero_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., isochromats, pools, 3)``.
        zero_matrix
            Cached no-RF system matrix for the current sequence execution, if available.
        zero_c
            Cached no-RF recovery vector for the current sequence execution, if available.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        if zero_matrix is None or zero_c is None:
            matrix, c = system_matrix(parameters, 0.0, 0.0, 0.0)
        else:
            matrix, c = zero_matrix, zero_c
        state = propagate(state, matrix, c, self._duration.to(state))
        return state, ()


class GradientBlock(BMCBlock):
    """Gradient-only block with position-dependent phase accrual."""

    def __init__(
        self,
        duration: torch.Tensor | float,
        positions: SpatialDimension[torch.Tensor],
        gradient_x: torch.Tensor | float = 0.0,
        gradient_y: torch.Tensor | float = 0.0,
        gradient_z: torch.Tensor | float = 0.0,
    ) -> None:
        """Initialize the block.

        Parameters
        ----------
        duration
            Duration in seconds.
        gradient_x
            Gradient amplitude in T/m along x.
        gradient_y
            Gradient amplitude in T/m along y.
        gradient_z
            Gradient amplitude in T/m along z.
        positions
            Isochromat positions in meters for ``(z, y, x)``.
        """
        super().__init__()
        self._duration = torch.as_tensor(duration)
        self.gradient_x = torch.as_tensor(gradient_x)
        self.gradient_y = torch.as_tensor(gradient_y)
        self.gradient_z = torch.as_tensor(gradient_z)
        self.positions = positions.apply(torch.as_tensor)
        self.n_iso = max(int(self.positions.z.numel()), int(self.positions.y.numel()), int(self.positions.x.numel()))
        if any(axis.numel() not in (1, self.n_iso) for axis in (self.positions.z, self.positions.y, self.positions.x)):
            raise ValueError(
                'positions.x, positions.y and positions.z must each have length 1 or match the isochromat count.'
            )

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self._duration

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        n_iso = state.shape[-3]
        if n_iso != self.n_iso:
            raise ValueError(
                f'Isochromat axis mismatch: state has {n_iso} isochromats but positions define {self.n_iso}.'
            )

        extra_off_resonance = gradient_to_extra_off_resonance(
            self.gradient_z,
            self.gradient_y,
            self.gradient_x,
            self.positions,
        )

        matrix = system_base_matrix(parameters, 0.0, extra_off_resonance)
        c = system_recovery_vector(parameters)
        c = c.unsqueeze(-2)
        duration = self._duration.unsqueeze(-1)
        state = apply_propagation_step(state, propagation_step(matrix, c, duration))
        return state, ()


class SpoilBlock(DelayBlock):
    """Perfect spoiling with non-zero duration."""

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
        *,
        zero_matrix: torch.Tensor | None = None,
        zero_c: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.
        zero_matrix
            Cached no-RF system matrix for the current sequence execution, if available.
        zero_c
            Cached no-RF recovery vector for the current sequence execution, if available.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        state, out = super().forward(parameters, state, zero_matrix=zero_matrix, zero_c=zero_c)
        mx, _, mz = state.unbind(-1)
        z = torch.zeros_like(mx)
        return torch.stack([z, z, mz], dim=-1), out


class AcquisitionBlock(BMCBlock):
    """Acquisition block that emits a readout."""

    def __init__(self, pool_index: int | None = None) -> None:
        """Initialize the block.

        Parameters
        ----------
        pool_index
            Pool index to read out. If ``None``, emit transverse signal for all pools.
        """
        super().__init__()
        self.pool_index = pool_index

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., pools, 3)``.

        Returns
        -------
        state
            State tensor. Shape ``(..., isochromats, pools, 3)``.
        """
        signal = transverse_readout(state)
        if self.pool_index is None:
            return state, (signal,)
        if not (0 <= self.pool_index < parameters.n_pools):
            raise ValueError('pool_index out of bounds.')
        return state, (signal[..., self.pool_index],)


class LongitudinalReadoutBlock(BMCBlock):
    """Read out longitudinal magnetization of a selected pool."""

    def __init__(self, pool_index: int = 0) -> None:
        """Initialize the block.

        Parameters
        ----------
        pool_index
            Pool index to read out.
        """
        super().__init__()
        self.pool_index = pool_index

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        if not (0 <= self.pool_index < parameters.n_pools):
            raise ValueError('pool_index out of bounds.')
        return state, (state[..., self.pool_index, 2].mean(dim=-1),)


class ResetBlock(BMCBlock):
    """Reset state to equilibrium or to a provided state."""

    def __init__(self, state: torch.Tensor | None = None) -> None:
        """Initialize the block.

        Parameters
        ----------
        state
            State tensor. Shape ``(..., pools, 3)``.
        """
        super().__init__()
        self.state = state

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.as_tensor(0.0)

    def forward(self, parameters: Parameters, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., isochromats, pools, 3)``.

        Returns
        -------
        state
            State tensor. Shape ``(..., isochromats, pools, 3)``.
        """
        if self.state is None:
            equilibrium = initial_state(parameters).to(state)
            batch = torch.broadcast_shapes(equilibrium.shape[:-3], state.shape[:-3])
            equilibrium = torch.broadcast_to(
                equilibrium,
                (*batch, state.shape[-3], equilibrium.shape[-2], equilibrium.shape[-1]),
            )
            return equilibrium.clone(), ()
        reset_state = self.state.to(state)
        if reset_state.ndim == state.ndim - 1:
            reset_state = reset_state.unsqueeze(-3)
        batch = torch.broadcast_shapes(reset_state.shape[:-3], state.shape[:-3])
        reset_state = torch.broadcast_to(
            reset_state,
            (*batch, state.shape[-3], reset_state.shape[-2], reset_state.shape[-1]),
        )
        return reset_state.clone(), ()


class BMCSequence(torch.nn.ModuleList, BMCBlock):
    """Sequence of Bloch-McConnell blocks."""

    def __init__(self, blocks: Sequence[BMCBlock] = ()) -> None:
        """Initialize the sequence.

        Parameters
        ----------
        blocks
            Sequence of Bloch-McConnell blocks.
        """
        torch.nn.ModuleList.__init__(self, blocks)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the sequence."""
        return sum(
            (b.duration for b in self if isinstance(b, BMCBlock)),
            start=torch.as_tensor(0.0),
        )

    def forward(
        self,
        parameters: Parameters,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the sequence of blocks.

        Parameters
        ----------
        parameters
            Simulation parameters.
        state
            State tensor. Shape ``(..., isochromats, pools, 3)``.

        Returns
        -------
        state
            State tensor. Shape ``(..., pools, 3)``.
        outputs
            List of output tensors.
        """
        parameters = parameters.to(state, copy=False)
        zero_matrix: torch.Tensor | None = None
        zero_c: torch.Tensor | None = None
        outputs: list[torch.Tensor] = []
        for block in self:
            assert isinstance(block, BMCBlock)  # noqa: S101
            if isinstance(block, DelayBlock | SpoilBlock):
                if zero_matrix is None or zero_c is None:
                    zero_matrix, zero_c = system_matrix(parameters, 0.0, 0.0, 0.0)
                state, out = block(parameters, state, zero_matrix=zero_matrix, zero_c=zero_c)
            else:
                state, out = block(parameters, state)
            outputs.extend(out)
        return state, tuple(outputs)
