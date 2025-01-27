"""Extended phase graph (EPG) signal models."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mrpro.data.MoveDataMixin import MoveDataMixin
from mrpro.operators.SignalModel import SignalModel


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
    ejp = torch.polar(torch.tensor(1.0), 1j * phase)
    inv_ejp = 1 / ejp

    return torch.stack(
        [
            cosa2 + 0j,
            ejp**2 * sina2,
            -1j * ejp * sina,
            inv_ejp**2 * sina2,
            cosa2 + 0j,
            1j * inv_ejp * sina,
            -1j / 2.0 * inv_ejp * sina,
            1j / 2.0 * ejp * sina,
            cosa + 0j,
        ],
        -1,
    ).reshape(*flip_angle.shape, 3, 3)


@torch.jit.script
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
    zero = state.new_zeros(*state.shape[:-2], 1)
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


class TensorAttributeMixin(torch.nn.Module):
    def __setattr__(self, name: str, value: Any):
        if isinstance(value, torch.Tensor):
            if value.is_leaf and value.requires_grad:
                self.register_parameter(name, torch.nn.Parameter(value))
            elif value.is_leaf:
                self.register_buffer(name, value)
            else:
                super().__setattr__(name, value)
        else:
            super().__setattr__(name, value)


@dataclass
class Parameters(MoveDataMixin):
    t1: torch.Tensor
    t2: torch.Tensor
    m0: torch.Tensor
    relative_b1: torch.Tensor | None = None

    @property
    def device(self) -> torch.device:
        device = super().device
        assert device is not None  # mypy
        return device

    @property
    def shape(self) -> torch.Size:
        shape = torch.broadcast_shapes(self.t1.shape, self.t2.shape, self.m0.shape)
        if self.relative_b1 is not None:
            shape = torch.broadcast_shapes(shape, self.relative_b1.shape)
        return shape

    @property
    def dtype(self) -> torch.dtype:
        dtype = torch.promote_types(self.t1.dtype, self.t2.dtype)
        dtype = torch.promote_types(dtype, self.m0.dtype)
        if self.relative_b1 is not None:
            dtype = torch.promote_types(dtype, self.relative_b1.dtype)
        return dtype


def acquisition(state: torch.Tensor, parameters: Parameters) -> torch.Tensor:
    return parameters.m0 * state[..., 0, 0]


def initial_state(
    shape: Sequence[int], n_sates: int = 20, device: torch.device | str = 'cpu', dtype: torch.dtype = torch.complex64
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
    if n_sates < 2:
        raise ValueError('Number of states should be at least 2.')
    return torch.zeros(*shape, 3, n_sates, device=device, dtype=dtype)


class EPGBlock(TensorAttributeMixin):
    """Base class for EPG blocks."""

    def __call__(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        return super().__call__(state, parameters)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, Sequence[torch.Tensor]]:
        raise NotImplementedError

    @property
    def duration(self) -> torch.Tensor:
        """Duration of the block in s."""
        raise NotImplementedError


class FispBlock(EPGBlock):
    """FISP data acquisition block."""

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
    ):
        try:
            self.flip_angles, self.rf_phases, self.te, self.tr = torch.broadcast_tensors(
                map(torch.as_tensor, (flip_angles, rf_phases, te, tr))
            )
        except RuntimeError:
            # Not broadcastable
            raise ValueError(
                f'Shapes of flip_angles ({flip_angles.shape}), rf_phases ({rf_phases.shape}), te ({te.shape}) and '
                f'tr ({tr.shape}) cannot be broadcasted.',
            ) from None

    @property
    def duration(self):
        return self.flip_angles.shape[-1] * self.tr

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        signal = []
        for flip_angle, rf_phase, te, tr in zip(self.flip_angles, self.rf_phases, self.te, self.tr, strict=True):
            state = rf(state, rf_matrix(flip_angle, rf_phase, parameters.relative_b1))
            state = relax(state, relax_matrix(te, parameters.t1, parameters.t2))
            signal.append(acquisition(state, parameters))
            state = gradient_dephasing(state)
            state = relax(state, relax_matrix((tr - te), parameters.t1, parameters.t2))
        return state, signal


class InversionBlock(EPGBlock):
    """T1 Inversion Preparation Block."""

    def __init__(self, inversion_time: torch.Tensor | float) -> None:
        self.inversion_time = torch.as_tensor(inversion_time)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        state = rf(state, rf_matrix(torch.pi, 0, parameters.relative_b1))
        state = relax(state, relax_matrix(self.inversion_time, parameters.t1, parameters.t2))
        return state, []

    @property
    def duration(self) -> torch.Tensor:
        return self.inversion_time


class T2PrepBlock(EPGBlock):
    """T2 Preparation Block."""

    def __init__(self, te: torch.Tensor | float) -> None:
        self.te = torch.as_tensor(te)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        # 90° pulse -> relaxation during TE/2 -> 180° pulse -> relaxation during TE/2 -> -90° pulse
        te2_relax = relax_matrix(self.te / 2, parameters.t1, parameters.t2)
        state = rf(state, rf_matrix(torch.pi / 2, 0, parameters.relative_b1))
        state = relax(state, te2_relax)
        state = rf(state, rf_matrix(torch.pi, torch.pi / 2, parameters.relative_b1))
        state = relax(state, te2_relax)
        state = rf(state, rf_matrix(torch.pi / 2, -torch.pi, parameters.relative_b1))
        # Spoiler
        state = gradient_dephasing(state)
        return state, []

    @property
    def duration(self) -> torch.Tensor:
        return self.te


class DelayBlock(EPGBlock):
    """Delay Block."""

    def __init__(self, delay_time: torch.Tensor | float) -> None:
        self.delay_time = torch.as_tensor(delay_time)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        state = relax(state, relax_matrix(self.delay_time, parameters.t1, parameters.t2))
        return state, []

    @property
    def duration(self):
        return self.delay_time


class EPGSequence(torch.nn.ModuleList, EPGBlock):
    """Sequene of EPG blocks.

    The sequence as multiple blocks, such as preparation pulses, acquisition blocks and delays.

    Classic MRF with a single inversion-pulse at the beginning followed by a train of RF pulses with different flip
    angles as described in Ma, D. et al. Magnetic resonance fingerprinting. Nature 495, 187-192 (2013)
    [http://dx.doi.org/10.1038/nature11971] can be simulated by the following sequence:
     - InversionBlock
     - FispBlock with flip_angles

    Cardiac MRF where a different preparation is done in each cardiac cycle followed by fixed number of RF pulses as
    described in Hamilton, J. I. et al. MR fingerprinting for rapid quantification of myocardial T1 , T2 , and proton
    spin density. Magn. Reson. Med. 77, 1446-1458 (2017) [http://doi.wiley.com/10.1002/mrm.26668]. It is a four-fold
    repetition of

                Block 0                   Block 1                   Block 2                     Block 3
       R-peak                   R-peak                    R-peak                    R-peak                    R-peak
    ---|-------------------------|-------------------------|-------------------------|-------------------------|-----

            [INV TI=20ms][ACQ]                     [ACQ]     [T2-prep TE=40ms][ACQ]    [T2-prep TE=80ms][ACQ]

    can be simulated as:
        - InversionBlock with inversion_time=20
        - FispBlock with 48 flip_angles
        - DelayBlock with delay_time=1000-(inversion_block.duration + fisp_block.duration)

        - FispBlock with 48 flip_angles
        - DelayBlock with delay_time=1000-(fisp_block.duration)

        - T2PrepBlock with te=40
        - FispBlock with 48 flip_angles
        - DelayBlock with delay_time=1000-(t2_prep_block.duration + fisp_block.duration)

        - T2PrepBlock with te=80
        - FispBlock with 48 flip_angles
        - DelayBlock with delay_time=1000-(t2_prep_block.duration + fisp_block.duration)

    """

    def __init__(self, blocks: Sequence[EPGBlock] = ()):
        torch.nn.ModuleList.__init__(self, blocks)

    def forward(self, state: torch.Tensor, parameters: Parameters) -> tuple[torch.Tensor, list[torch.Tensor]]:
        signals: list[torch.Tensor] = []
        block: EPGBlock
        for block in self:
            state, signal = block(state, parameters)
            signals.extend(signal)
        return state, signals

    @property
    def duration(self) -> torch.Tensor:
        return sum(block.duration() for block in self)


class EPGSignalModel(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]):
    """EPG signal model."""

    def __init__(self, sequence: EPGSequence | Sequence[EPGBlock], n_states: int = 20):
        self.n_states = n_states
        if not isinstance(sequence, EPGSequence):
            self.sequence = EPGSequence(sequence)
        else:
            self.sequence = sequence

    def forward(
        self, t1: torch.Tensor, t2: torch.Tensor, m0: torch.Tensor, relative_b1: torch.Tensor | None = None
    ) -> tuple[torch.Tensor]:
        parameters = Parameters(t1, t2, m0, relative_b1)
        state = initial_state(parameters.shape, n_sates=self.n_states, device=parameters.device, dtype=parameters.dtype)
        _, signals = self.sequence(state, parameters)
        signal = torch.stack(list(signals), dim=0)
        return (signal,)


class CardiacFingerprinting(SignalModel[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Cardiac MR Fingerprinting signal model."""

    def __init__(self, acquisition_times: torch.Tensor, te: float) -> None:
        sequence = EPGSequence()
        max_flip_angles = [12.5, 18.75, 25.0, 25.0, 25.0, 12.5, 18.75, 25.0, 25.0, 25.0, 12.5, 18.75, 25.0, 25.0, 25.0]
        for i in range(len(acquisition_times)):
            block = EPGSequence()
            match i % 5:
                case 0:
                    block.append(InversionBlock(0.020))
                case 1:
                    pass
                case 2:
                    block.append(T2PrepBlock(0.03))
                case 3:
                    block.append(T2PrepBlock(0.05))
                case 4:
                    block.append(T2PrepBlock(0.1))
            flip_angles = torch.cat((torch.linspace(4, max_flip_angles[i], 16), torch.full((31,), max_flip_angles[i])))
            block.append(FispBlock(flip_angles, 0.0, tr=0.01, te=te))
            if i > 0:
                delay = (acquisition_times[i] - acquisition_times[i - 1]) - block.duration
                sequence.append(DelayBlock(delay))
            sequence.append(block)
        self.model = EPGSignalModel(sequence, n_states=20)

    def forward(self, t1: torch.Tensor, t2: torch.Tensor, m0: torch.Tensor) -> tuple[torch.Tensor]:
        return self.model(t1, t2, m0, None)
