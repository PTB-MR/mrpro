"""Extended phase graph (EPG) signal models."""

# Based on https://github.com/fzimmermann89/epgtorch/
# Copyright (c) 2022 Felix Zimmermann, MIT License

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.operators.SignalModel import SignalModel
from mrpro.utils.reshape import unsqueeze_tensors_right
from mrpro.utils.TensorAttributeMixin import TensorAttributeMixin


@dataclass
class Parameters(MoveDataMixin):
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
        Phase of the RF pulse
    relative_b1
        Scaling of flip angle due to B1 inhomogeneities

    Returns
    -------
        Matrix describing the mixing of the EPG configuration states due to an RF pulse.
    """
    if relative_b1 is not None:
        flip_angle = flip_angle * relative_b1[None, ...].abs()
        phase = phase + relative_b1[None, ...].angle()
    cosa = torch.cos(flip_angle)
    sina = torch.sin(flip_angle)
    cosa2 = (cosa + 1) / 2
    sina2 = 1 - cosa2
    ejp = torch.polar(torch.tensor(1.0), phase)
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
        EPG configuration states. Shape `..., 3 (f_plus, f_minus, z), n `
    matrix
        Rotation matrix describing the mixing of the EPG configuration states due to an RF pulse.
        Shape `..., 3, 3`

    Returns
    -------
        EPG configuration states after RF pulse. Shape `..., 3, n`
    """
    return matrix @ state


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
        EPG configuration states after gradient. Shape `..., 3, n`
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
    return torch.stack([e2, e2, e1], dim=-1)


@torch.jit.script
def relax(state: torch.Tensor, relaxation_matrix: torch.Tensor, t1_recovery: bool = True) -> torch.Tensor:
    """Propagate EPG states through a period of relaxation and recovery.

    Parameters
    ----------
    state
        EPG configuration states Fplus, Fminus, Z
    relaxation_matrix
        matrix describing EPG relaxation
    t1_recovery
        recovery of longitudinal EPG states

    Returns
    -------
        EPG configuration states after relaxation and recovery
    """
    state = relaxation_matrix[..., None] * state

    if t1_recovery:
        state[..., 2, 0] = state[..., 2, 0] + (1 - relaxation_matrix[..., -1])
    return state


def acquisition(state: torch.Tensor, m0: torch.Tensor) -> torch.Tensor:
    """Calculate the signal from the EPG state."""
    return m0 * state[..., 0, 0]


def initial_state(
    shape: Sequence[int], n_states: int = 20, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.complex64
) -> torch.Tensor:
    """Generate initial EPG state.

    Parameters
    ----------
    shape
        Shape of the state tensor, excluding the last two dimensions. Shouöd match the parameters shape.
    n_states
        Number of EPG configuration states.
    device
        Device to create the tensor on.
    dtype
        Data type of the tensor.

    Returns
    -------
        Initial EPG state tensor. Shape `*shape,3, n_state)`
    """
    if n_states < 2:
        raise ValueError('Number of states should be at least 2.')
    state = torch.zeros(*shape, 3, n_states, device=device, dtype=dtype)
    # Start in equilibrium state
    state[..., :, 0] = torch.tensor((0.0, 0, 1.0))
    return state


class EPGBlock(TensorAttributeMixin, ABC):
    """Base class for EPG blocks."""

    def __call__(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        """Apply the block."""
        return super().__call__(state, parameters)

    @abstractmethod
    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        """Apply the block."""
        raise NotImplementedError

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        raise NotImplementedError


class RFBlock(EPGBlock):
    """RF pulse block."""

    def __init__(self, flip_angle: torch.Tensor | float, phase: torch.Tensor | float) -> None:
        """Initialize RF block.

        Parameters
        ----------
        flip_angle
            flip angle [rad]
        phase
            initial rf phase
        """
        super().__init__()
        self.flip_angle = torch.as_tensor(flip_angle)
        self.phase = torch.as_tensor(phase)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the RF to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        matrix = rf_matrix(self.flip_angle, self.phase, parameters.relative_b1)
        return rf(state, matrix), []

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.tensor(0.0)


class AcquisitionBlock(EPGBlock):
    """Acquisition block."""

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the acquisition block to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states (unchanged) after the block and the acquired signal
        """
        return state, [acquisition(state, parameters.m0)]

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return torch.tensor(0.0)


class GradientDephasingBlock(EPGBlock):
    """Gradient dephasing block."""

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:  # noqa: ARG002
        """Apply the gradient dephasing block to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        return gradient_dephasing(state), []

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

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self.tr.sum(0)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the FISP block to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

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
            state = rf(state, rf_matrix(flip_angle, rf_phase, parameters.relative_b1))
            state = relax(state, relax_matrix(te, parameters.t1, parameters.t2))
            signal.append(acquisition(state, parameters.m0))
            state = gradient_dephasing(state)
            state = relax(state, relax_matrix((tr - te), parameters.t1, parameters.t2))
        return state, signal


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

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the inversion block to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        state = rf(state, rf_matrix(torch.tensor(torch.pi), torch.tensor(0.0)))
        state = relax(state, relax_matrix(self.inversion_time, parameters.t1, parameters.t2))
        return state, []

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

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the T2 preparation block to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        # 90° pulse -> relaxation during TE/2 -> 180° pulse -> relaxation during TE/2 -> -90° pulse
        te2_relax = relax_matrix(self.te / 2, parameters.t1, parameters.t2)
        state = rf(state, rf_matrix(torch.tensor(torch.pi / 2), torch.tensor(0.0)))
        state = relax(state, te2_relax)
        state = rf(state, rf_matrix(torch.tensor(torch.pi), torch.tensor(torch.pi / 2)))
        state = relax(state, te2_relax)
        state = rf(state, rf_matrix(torch.tensor(torch.pi / 2), -torch.tensor(torch.pi)))
        # Spoiler
        state = gradient_dephasing(state)
        return state, []

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

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the delay block to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states after the block and an empty list
        """
        (delay_time,) = unsqueeze_tensors_right(self.delay_time, ndim=parameters.ndim + 1)  # +1 for time dimension
        state = relax(state, relax_matrix(delay_time, parameters.t1, parameters.t2))
        return state, []

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return self.delay_time


class EPGSequence(torch.nn.ModuleList, EPGBlock):
    """Sequene of EPG blocks.

    The sequence as multiple blocks, such as preparation pulses, acquisition blocks and delays.

    Classic MRF with a single inversion-pulse at the beginning followed by a train of RF pulses with different flip
    angles as described in [MA2013]_ Ma, D. et al. Magnetic resonance fingerprinting. Nature 495, 187-192 (2013)
    [http://dx.doi.org/10.1038/nature11971] can be simulated by the following sequence:
     - InversionBlock
     - FispBlock with flip_angles

    Cardiac MRF where a different preparation is done in each cardiac cycle followed by fixed number of RF pulses as
    described in [HAMI2017]_. It is a four-fold
    repetition of

                Block 0                   Block 1                   Block 2                     Block 3
       R-peak                   R-peak                    R-peak                    R-peak                    R-peak
    ---|-------------------------|-------------------------|-------------------------|-------------------------|-----

            [INV TI=30ms][ACQ]                     [ACQ]     [T2-prep TE=50ms][ACQ]    [T2-prep TE=100ms][ACQ]

    can be simulated as:
        - InversionBlock with inversion_time=30
        - FispBlock with 47 flip_angles
        - DelayBlock with delay_time=1000-(inversion_block.duration + fisp_block.duration)

        - FispBlock with 47 flip_angles
        - DelayBlock with delay_time=1000-(fisp_block.duration)

        - T2PrepBlock with te=50
        - FispBlock with 47 flip_angles
        - DelayBlock with delay_time=1000-(t2_prep_block.duration + fisp_block.duration)

        - T2PrepBlock with te=100
        - FispBlock with 47 flip_angles
        - DelayBlock with delay_time=1000-(t2_prep_block.duration + fisp_block.duration)

    See also `CardiacFingerprinting` for an implementation.

    References
    ----------
    .. [MA2013] Ma, D et al.(2013) Magnetic resonance fingerprinting. Nature 495 http://dx.doi.org/10.1038/nature11971
    .. [HAMI2017] Hamilton, J. I. et al. (2017) MR fingerprinting for rapid quantification of myocardial T1, T2, and
            proton spin density. Magn. Reson. Med. 77 http://doi.wiley.com/10.1002/mrm.26668

    """

    def __init__(self, blocks: Sequence[EPGBlock] = ()) -> None:
        """Initialize an EPG Sequence.

        Parameters
        ----------
        blocks
            blocks such as RF, delays, acquisitions etc.
        """
        torch.nn.ModuleList.__init__(self, blocks)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Apply the sequence of blocks to the EPG state.

        Parameters
        ----------
        state
            EPG configuration states
        parameters
            Tissue parameters

        Returns
        -------
            EPG configuration states after the sequence of blocks and the acquired signals
        """
        signals: list[torch.Tensor] = []
        block: EPGBlock
        for block in self:
            state, signal = block(state, parameters)
            signals.extend(signal)
        return state, signals

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block."""
        return sum(block.duration for block in self)


class EPGSignalModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]):
    """EPG signal model."""

    def __init__(self, sequence: EPGSequence | Sequence[EPGBlock], n_states: int = 20):
        """Initialize the EPG signal model.

        Parameters
        ----------
        sequence
            EPG sequence of blocks
        n_states
            Number of EPG configuration states. This model uses a fixed number of states for performance reasons.
            Should be large enough to capture the signal dynamics.
            More states increase the accuracy of the simulation but also the computational cost.
        """
        super().__init__()
        self.n_states = n_states
        if not isinstance(sequence, EPGSequence):
            self.sequence = EPGSequence(sequence)
        else:
            self.sequence = sequence

    def forward(
        self, m0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, relative_b1: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        """Simulate the EPG signal.

        Parameters
        ----------
        m0
            Steady state magnetization (complex)
        t1
            T1 relaxation time
        t2
            T2 relaxation time
        relative_b1
            Relative B1 scaling factor (complex)

        Returns
        -------
            Simulated EPG signal with the different acquisitions in the first dimension.
        """
        parameters = Parameters(m0, t1, t2, relative_b1)
        state = initial_state(
            parameters.shape, n_states=self.n_states, device=parameters.device, dtype=parameters.dtype.to_complex()
        )
        _, signals = self.sequence(state, parameters)
        signal = torch.stack(list(signals), dim=0)
        return (signal,)


__all__ = [
    'AcquisitionBlock',
    'DelayBlock',
    'EPGSequence',
    'EPGSignalModel',
    'FispBlock',
    'GradientDephasingBlock',
    'InversionBlock',
    'Parameters',
    'RFBlock',
    'T2PrepBlock',
    'initial_state',
]
