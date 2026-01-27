"""Time-segmented Fast Fourier Transform operator."""

from __future__ import annotations

import torch

from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator


class TimeSegmentedFastFourierOp(LinearOperator):
    """Time-segmented FFT operator for B0 inhomogeneity compensation.

    This operator approximates the off-resonance effect using a time-segmented approach.
    It decomposes the phase accumulation into a sum of L segments, where the temporal
    component is handled by an interpolator and the spatial component by basis maps.
    """

    _basis: torch.Tensor
    _interpolator: torch.Tensor
    _fft_op: FastFourierOp
    _n_segs: int
    _n_ro: int

    def __init__(
        self,
        b0_map: torch.Tensor,
        readout_times: torch.Tensor,
        num_segments: int,
        num_frequencies: int = -1,
    ) -> None:
        """Initialize TimeSegmentedFFTOp.

        Parameters
        ----------
        b0_map
            Field map B0 in Hz. Expected shape (..., Z, Y, X).
        readout_times
            Readout time vector in seconds. Expected shape (X,).
        num_segments
            Number of time segments L.
        num_frequencies
            Number of frequencies to use when solver_reduce_memory is True.
        """
        super().__init__()

        self._n_segs = num_segments
        self._n_ro = readout_times.shape[-1]

        if self._n_ro != b0_map.shape[-1]:
            raise ValueError('readout_times must have the same size as the last dimension of field_map')

        # Get flattened 1D field map in rad/s
        w0_flat = b0_map.flatten() * 2 * torch.pi

        if num_frequencies > 0:
            # Quantile-based sampling: Pick certain number of frequencies based on the data distribution
            q_steps = torch.linspace(0, 1, num_frequencies, device=b0_map.device)
            w_design = torch.quantile(w0_flat, q_steps)
        else:
            # Use every single frequency in the map.
            w_design = w0_flat

        t_ro = readout_times
        t_seg = torch.linspace(t_ro[0], t_ro[-1], num_segments, device=b0_map.device)

        # Solve for Interpolator C: Solve the system A @ C = B in a least-squares sense.
        # A (Segment Phase): exp(-i * w * t_seg)
        # B (Readout Phase): exp(-i * w * t_ro)

        # Segmented phase values from segmented time vector with shape (N_freqs, L)
        phase_segments = torch.exp(-1j * (w_design[:, None] @ t_seg[None, :]))
        # Target phase values from readout times with shape (N_freqs, N_ro)
        phase_readout = torch.exp(-1j * (w_design[:, None] @ t_ro[None, :]))
        # Interpolator obtained by pinv(phase_segments) @ phase_readout with shape (L, N_ro)
        interpolator = torch.linalg.lstsq(phase_segments, phase_readout, rcond=1e-15).solution

        # Precompute spatial basis functions B_l(r) = exp(-i * w(r) * t_seg_l)
        # with shape (L, ..., Z, Y, X) where ... are the spatial dimensions of b0_map.
        basis = torch.exp(-1j * 2 * torch.pi * torch.einsum('l, ... -> l...', t_seg, b0_map))

        # Register buffers
        self.register_buffer('_basis', basis)  # (L, Z, Y, X)
        self.register_buffer('_interpolator', interpolator)  # (L, N_ro)

        # mrpro fast fourier operator
        self._fft_op = FastFourierOp()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward operator: y = sum_l C_l * FFT(B_l * x)."""
        # x shape: (Batch..., Z, Y, X)

        # Multiply Input with Basis Maps
        # ...zyx    Input x (Batch, Z, Y, X)
        # lzyx      Basis (L, Z, Y, X)
        # l...zyx   Result (L, Batch, Z, Y, X)
        x_seg = torch.einsum('...zyx, lzyx -> l...zyx', x, self._basis)

        # FFT (applied to L and Batch dimensions independently)
        (ksp_seg,) = self._fft_op(x_seg)

        # Multiply with Interpolator and Sum
        # l...x   K-space segment (L, Batch, Z, Y, X_readout)
        # lx      Interpolator (L, X_readout)
        # ...x    Result (Batch, Z, Y, X_readout) - Summed over L
        res = torch.einsum('l...x, lx -> ...x', ksp_seg, self._interpolator)

        return (res,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply adjoint operator: x = sum_l B_l_conj * IFFT(C_l_conj * y)."""
        # y shape: (Batch..., Z, Y, X_kspace)

        # Multiply K-space with Conjugate Interpolator
        # ...x      Input y (Batch, Z, Y, X)
        # lx        Interpolator (L, X)
        # l...x     Result (L, Batch, Z, Y, X)
        y_seg = torch.einsum('...x, lx -> l...x', y, self._interpolator.conj())

        # Adjoint FFT
        (img_seg,) = self._fft_op.adjoint(y_seg)

        # Multiply with Conjugate Basis and Sum
        # l...zyx   Image segment (L, Batch, Z, Y, X)
        # lzyx      Basis (L, Z, Y, X)
        # ...zyx    Result (Batch, Z, Y, X) - Summed over L
        res = torch.einsum('l...zyx, lzyx -> ...zyx', img_seg, self._basis.conj())

        return (res,)
