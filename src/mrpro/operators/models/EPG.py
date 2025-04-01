"""Extended phase graph (EPG) signal models."""

# Based on https://github.com/fzimmermann89/epgtorch/
# Copyright (c) 2022 Felix Zimmermann, MIT License

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch

from mrpro.data.Dataclass import Dataclass
from mrpro.utils.reshape import unsqueeze_tensors_right
from mrpro.utils.TensorAttributeMixin import TensorAttributeMixin


class Parameters(Dataclass):
    """Tissue parameters for EPG simulation."""

    m0: torch.Tensor
    """Steay state magnetization (complex)"""

    t1: torch.Tensor
    """T1 relaxation time [s]"""

    t2: torch.Tensor
    """T2 relaxation time [s]"""

    relative_b1: torch.Tensor | None = None
    """Relative B1 scaling factor (complex)"""

    @property
    def device(self) -> torch.device:
        """Device of the parameters."""
        device = super().device
        assert device is not None  # mypy # noqa: S101
        return device

    @property
    def shape(self) -> torch.Size:
        """Broadcasted shape of the parameters."""
        shape = torch.broadcast_shapes(self.t1.shape, self.t2.shape, self.m0.shape)
        if self.relative_b1 is not None:
            shape = torch.broadcast_shapes(shape, self.relative_b1.shape)
        return shape

    @property
    def dtype(self) -> torch.dtype:
        """Promoted data type of the parameters."""
        dtype = torch.promote_types(self.t1.dtype, self.t2.dtype)
        dtype = torch.promote_types(dtype, self.m0.dtype)
        if self.relative_b1 is not None:
            dtype = torch.promote_types(dtype, self.relative_b1.dtype)
        return dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of the parameters."""
        ndim = max(self.t1.ndim, self.t2.ndim, self.m0.ndim)
        if self.relative_b1 is not None:
            ndim = max(ndim, self.relative_b1.ndim)
        return ndim


@torch.jit.script
def rf_matrix(
    flip_angle: torch.Tensor,
    phase: torch.Tensor,
    relative_b1: torch.Tensor | None = None,
) -> torch.Tensor:
    """Initialize the rotation matrix describing the RF pulse.

    Parameters
    ----------
    flip_angle
        Flip angle of the RF pulse in rad
    phase
        Phase of the RF pulse in rad
    relative_b1
        Scaling of flip angle due to B1 inhomogeneities

    Returns
    -------
        Matrix describing the mixing of the EPG configuration states due to an RF pulse.
    """
    if relative_b1 is not None:
        # relative_b1 and flip_angle need to be correctly broadcasted at this point
        flip_angle = flip_angle * relative_b1.abs()
        phase = phase + relative_b1.angle()
    cosa = torch.cos(flip_angle)
    sina = torch.sin(flip_angle)
    cosa2 = (cosa + 1) / 2
    sina2 = 1 - cosa2
    ejp = torch.polar(torch.ones_like(phase), phase)
    inv_ejp = 1 / ejp
    new_shape = flip_angle.shape + (3, 3)  # noqa: RUF005 # not supported in torchscript

    return torch.stack(
        [
            cosa2 + 0.0j,
            ejp**2 * sina2,
            -1.0j * ejp * sina,
            inv_ejp**2 * sina2,
            cosa2 + 0.0j,
            1.0j * inv_ejp * sina,
            -0.5j * inv_ejp * sina,
            0.5j * ejp * sina,
            cosa + 0.0j,
        ],
        -1,
    ).reshape(new_shape)


def rf(state: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Propagate EPG states through an RF rotation.

    Parameters
    ----------
    state
        EPG configuration states. Shape `..., 3 (f_plus, f_minus, z), n`
    matrix
        Rotation matrix describing the mixing of the EPG configuration states due to an RF pulse.
        Shape `..., 3, 3`

    Returns
    -------
        EPG configuration states after RF pulse. Shape `..., 3 (f_plus, f_minus, z), n`
    """
    return matrix.to(state) @ state


@torch.jit.script
def gradient_dephasing(state: torch.Tensor) -> torch.Tensor:
    """Propagate EPG states through a "unit" gradient.

    Parameters
    ----------
    state
        EPG configuration states. Shape `..., 3 (f_plus, f_minus, z), n`
        with n being the number of configuration states > 1

    Returns
    -------
        EPG configuration states after gradient. Shape `..., 3 (f_plus, f_minus, z), n`
    """
    zero = state.new_zeros(state.shape[:-2] + (1,))
    f_plus = torch.cat((state[..., 1, 1:2].conj(), state[..., 0, :-1]), dim=-1)
    f_minus = torch.cat((state[..., 1, 1:], zero), -1)
    z = state[..., 2, :]
    return torch.stack((f_plus, f_minus, z), -2)


@torch.jit.script
def relax_matrix(relaxation_time: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Calculate relaxation vector.

    Parameters
    ----------
    relaxation_time
        relaxation time
    t1
        longitudinal relaxation time
    t2
        transversal relaxation time

    Returns
    -------
        vector describing relaxation
    """
    e2 = torch.exp(-relaxation_time / t2)
    e1 = torch.exp(-relaxation_time / t1)
    e1, e2 = torch.broadcast_tensors(e1, e2)
    return torch.stack([e2, e2, e1], dim=-1)


@torch.jit.script
def relax(states: torch.Tensor, relaxation_matrix: torch.Tensor, t1_recovery: bool = True) -> torch.Tensor:
    """Propagate EPG states through a period of relaxation and recovery.

    Parameters
    ----------
    states
        EPG configuration states. Shape `..., 3 (f_plus, f_minus, z), n`
    relaxation_matrix
        matrix describing EPG relaxation
    t1_recovery
        recovery of longitudinal EPG states

    Returns
    -------
        EPG configuration states after relaxation and recovery.
        Shape `..., 3 (f_plus, f_minus, z), n`
    """
    relaxation_matrix = relaxation_matrix.to(states)
    states = relaxation_matrix[..., None] * states

    if t1_recovery:
        states[..., 2, 0] = states[..., 2, 0] + (1 - relaxation_matrix[..., -1])
    return states


def acquisition(state: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
    """Calculate the acquired signal from the EPG state."""
    return m0 * state[..., 0, 0]


def initial_state(
    shape: Sequence[int], n_states: int = 20, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.complex64
) -> torch.Tensor:
    """Generate initial EPG states.

    Parameters
    ----------
    shape
        Shape of the state tensor, excluding the last two dimensions. Should match the parameters shape.
    n_states
        Number of EPG configuration states.
    device
        Device to create the state tensor on.
    dtype
        Data type of the state tensor.

    Returns
    -------
        Initial EPG state tensor. Shape `*shape, 3 (f_plus, f_minus, z), n`
    """
    if n_states < 2:
        raise ValueError('Number of states should be at least 2.')
    states = torch.zeros(*shape, 3, n_states, device=device, dtype=dtype)
    # Start in equilibrium state
    states[..., :, 0] = torch.tensor((0.0, 0, 1.0))
    return states


class EPGBlock(TensorAttributeMixin, ABC):
    """Base class for EPG blocks."""

    def __call__(
        self, parameters: Parameters, states: torch.Tensor | int = 20
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states.
            If an integer value, the equilibrium state (0, 0, 1) will be initialized with
            the given number of EPG configuration states. The number should be large enough to capture the
            signal dynamics. More states increase the accuracy of the simulation but also the computational cost.
        """
        if isinstance(states, int):
            states = initial_state(
                parameters.shape, n_states=states, device=parameters.device, dtype=parameters.dtype.to_complex()
            )
        return super().__call__(parameters, states)

    @abstractmethod
    def forward(self, parameters: Parameters, states: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the block."""
        raise NotImplementedError

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        raise NotImplementedError


class RFBlock(EPGBlock):
    """RF pulse block."""

    def __init__(self, flip_angle: torch.Tensor | float, phase: torch.Tensor | float) -> None:
        """Initialize RF pulse block.

        Parameters
        ----------
        flip_angle
            flip angle in rad
        phase
            initial rf phase in rad
        """
        super().__init__()
        self.flip_angle = torch.as_tensor(flip_angle)
        self.phase = torch.as_tensor(phase)

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the RF pulse to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        matrix = rf_matrix(self.flip_angle, self.phase, parameters.relative_b1)
        return rf(states, matrix), ()

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.tensor(0.0)


class AcquisitionBlock(EPGBlock):
    """Acquisition block."""

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the acquisition block to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states

        Returns
        -------
            EPG configuration states (unchanged) after the block and the acquired signal
        """
        return states, (acquisition(states, parameters.m0),)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.tensor(0.0)


class GradientDephasingBlock(EPGBlock):
    """Gradient dephasing block."""

    def forward(
        self,
        parameters: Parameters,  # noqa: ARG002
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the gradient dephasing block to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states


        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        return gradient_dephasing(states), ()

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.tensor(0.0)


class FispBlock(EPGBlock):
    """FISP data acquisition block.

    The block consists of a series of RF pulses with different flip angles and phases followed by an acquisition:
        - rf with flip_angle[i] and rf_phase[i]
        - relax for te[i]
        - acquisition
        - gradient_dephasing
        - relax for tr[i] - te[i]
    """

    flip_angles: torch.Tensor
    rf_phases: torch.Tensor
    te: torch.Tensor
    tr: torch.Tensor

    def __init__(
        self,
        flip_angles: torch.Tensor | float,
        rf_phases: torch.Tensor | float,
        te: torch.Tensor | float,
        tr: torch.Tensor | float,
    ) -> None:
        """Initialize the FISP block.

        Parameters
        ----------
        flip_angles
            Flip angles of the RF pulses in rad
        rf_phases
            Phase of the RF pulses in rad
        te
            Echo time
        tr
            Repetition time
        """
        super().__init__()

        flip_angles_, rf_phases_, te_, tr_ = map(torch.as_tensor, (flip_angles, rf_phases, te, tr))
        try:
            self.flip_angles, self.rf_phases, self.te, self.tr = torch.broadcast_tensors(
                flip_angles_, rf_phases_, te_, tr_
            )
        except RuntimeError:
            raise ValueError(
                f'Shapes of flip_angles ({flip_angles_.shape}), rf_phases ({rf_phases_.shape}), te ({te_.shape}) and '
                f'tr ({tr_.shape}) cannot be broadcasted.',
            ) from None
        if (self.te > self.tr).any():
            raise ValueError(f'echotime ({self.te}) should be smaller than repetition time ({self.tr}).')
        if (self.te < 0).any():
            raise ValueError(f'Negative echo time ({self.te.amin()}) not allowed.')

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self.tr.sum(0)

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the FISP block to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states

        Returns
        -------
            EPG configuration states after the block and the acquired signals
        """
        signal = []
        # +1 for time dimension
        unsqueezed = unsqueeze_tensors_right(
            self.flip_angles, self.rf_phases, self.te, self.tr, ndim=parameters.ndim + 1
        )
        for flip_angle, rf_phase, te, tr in zip(*unsqueezed, strict=True):
            states = rf(states, rf_matrix(flip_angle, rf_phase, parameters.relative_b1))
            states = relax(states, relax_matrix(te, parameters.t1, parameters.t2))
            signal.append(acquisition(states, parameters.m0))
            states = gradient_dephasing(states)
            states = relax(states, relax_matrix((tr - te), parameters.t1, parameters.t2))
        return states, tuple(signal)


class InversionBlock(EPGBlock):
    """T1 Inversion Preparation Block.

    The inversion pulse is assumed to be B1 insensitive.
    """

    def __init__(self, inversion_time: torch.Tensor | float) -> None:
        """Initialize the inversion block.

        Parameters
        ----------
        inversion_time
            Inversion time
        """
        super().__init__()

        self.inversion_time = torch.as_tensor(inversion_time)
        if (self.inversion_time < 0).any():
            raise ValueError(f'Negative inversion time ({self.inversion_time.amin()}) not allowed.')

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the inversion block to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        states = rf(states, rf_matrix(torch.tensor(torch.pi), torch.tensor(0.0)))
        states = relax(states, relax_matrix(self.inversion_time, parameters.t1, parameters.t2))
        return states, ()

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self.inversion_time


class T2PrepBlock(EPGBlock):
    """T2 Preparation Block.

    Consists of a 90° pulse, relaxation during TE/2, 180° pulse, relaxation during TE/2 and a -90° pulse.

    The pulses are assumed to be B1 insensitive.
    """

    def __init__(self, te: torch.Tensor | float) -> None:
        """Initialize the T2 preparation block.

        Parameters
        ----------
        te
            Echo time of the T2 preparation block
        """
        super().__init__()

        self.te = torch.as_tensor(te)
        if (self.te < 0).any():
            raise ValueError(f'Negative echo time ({self.te.amin()}) not allowed.')

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the T2 preparation block to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        # 90° pulse -> relaxation during TE/2 -> 180° pulse -> relaxation during TE/2 -> -90° pulse
        te2_relax = relax_matrix(self.te / 2, parameters.t1, parameters.t2)
        states = rf(states, rf_matrix(torch.tensor(torch.pi / 2), torch.tensor(0.0)))
        states = relax(states, te2_relax)
        states = rf(states, rf_matrix(torch.tensor(torch.pi), torch.tensor(torch.pi / 2)))
        states = relax(states, te2_relax)
        states = rf(states, rf_matrix(torch.tensor(torch.pi / 2), -torch.tensor(torch.pi)))
        # Spoiler
        states = gradient_dephasing(states)
        return states, ()

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self.te


class DelayBlock(EPGBlock):
    """Delay Block."""

    def __init__(self, delay_time: torch.Tensor | float) -> None:
        """Initialize the delay block.

        Parameters
        ----------
        delay_time
            Delay time
        """
        super().__init__()
        self.delay_time = torch.as_tensor(delay_time)
        if (self.delay_time < 0).any():
            raise ValueError(f'Negative delay time ({self.delay_time.amin()})')

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the delay block to the EPG state.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        (delay_time,) = unsqueeze_tensors_right(self.delay_time, ndim=parameters.ndim)
        states = relax(states, relax_matrix(delay_time, parameters.t1, parameters.t2))
        return states, ()

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self.delay_time


class EPGSequence(torch.nn.ModuleList, EPGBlock):
    """Sequene of EPG blocks.

    A sequence as multiple blocks, such as preparation pulses, acquisition blocks and delays.

    Classic MRF with a single inversion-pulse at the beginning followed by a train of RF pulses with different flip
    angles as described in [MA2013]_ can be simulated by the following sequence:
     - InversionBlock
     - FispBlock with varying flip_angles

        You can also nest `EPGSequence`, i.e., use it as a block in another `EPGSequence`. This allows to describe
        complex MR acquisitions with multiple repetitions of the same blocks.


    References
    ----------
    .. [MA2013] Ma, D et al.(2013) Magnetic resonance fingerprinting. Nature 495 http://dx.doi.org/10.1038/nature11971


    """

    def __init__(self, blocks: Sequence[EPGBlock] = ()) -> None:
        """Initialize an EPG Sequence.

        Parameters
        ----------
        blocks
            blocks such as RF, delays, acquisitions etc.
        """
        torch.nn.ModuleList.__init__(self, blocks)

    def forward(
        self,
        parameters: Parameters,
        states: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """Apply the sequence of blocks to the EPG states.

        Parameters
        ----------
        parameters
            Tissue parameters
        states
            EPG configuration states.

        Returns
        -------
            EPG configuration states after the sequence of blocks and the acquired signals
        """
        signals: list[torch.Tensor] = []
        block: EPGBlock
        for block in self:
            states, signal = block(parameters, states)
            signals.extend(signal)
        return states, tuple(signals)

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return sum(block.duration for block in self)


__all__ = [
    'AcquisitionBlock',
    'DelayBlock',
    'EPGSequence',
    'FispBlock',
    'GradientDephasingBlock',
    'InversionBlock',
    'Parameters',
    'RFBlock',
    'T2PrepBlock',
    'initial_state',
]
