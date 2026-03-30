"""Conjugate Phase Fast Fourier Transform operator."""

import torch

from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator


class ConjugatePhaseFastFourierOp(LinearOperator):
    """Optimized 3D Conjugate Phase Reconstruction (CPR) Operator."""

    unique_freqs: torch.Tensor
    masks: torch.Tensor
    phasor: torch.Tensor
    _fft_op: FastFourierOp

    def __init__(self, b0_map: torch.Tensor, readout_times: torch.Tensor) -> None:
        """
        Initialize conjugate phase operator.

        Args:
            b0_map (torch.Tensor): Off-resonance map in Hz. Shape (Z, Y, X).
            readout_times (torch.Tensor): Readout time vector in seconds. Shape (X,).
        """
        super().__init__()

        # Identify Unique Frequencies
        self.unique_freqs, _ = torch.unique(b0_map, return_inverse=True)

        # Reshape freqs for broadcasting against map: (L) -> (L, 1, 1, ...)
        freqs_bc = self.unique_freqs.view(-1, *([1] * b0_map.ndim))

        # Create Masks: Result (L, ..., Z, Y, X)
        self.masks = (b0_map.unsqueeze(0) == freqs_bc).type(torch.complex64)

        # Calculate Phase: 2 * pi * f * t
        # Compute the outer product of Freqs (L) and Time (X) using einsum (result Shape: (L, X))
        phase_rad = 2 * torch.pi * torch.einsum('l, x -> lx', self.unique_freqs, readout_times)

        # Store Phasor (L, X)
        self.register_buffer('phasor', torch.exp(-1j * phase_rad))

        self._fft_op = FastFourierOp()

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor,]:
        """Image (..., Z, Y, X) -> k-space (..., Z, Y, X)."""
        # Spatial Segmentation: (L, ..., Z, Y, X)
        x_seg = torch.einsum('l...zyx, ...zyx -> l...zyx', self.masks, image)

        # FFT applied to each segment
        (k_seg,) = self._fft_op(x_seg)

        # Multiply with Phasor and sum over segments: (L, ..., Z, Y, X) * (L, X) -> (..., Z, Y, X)
        res = torch.einsum('l...zyx, lx -> ...zyx', k_seg, self.phasor)

        return (res,)

    def adjoint(self, kspace: torch.Tensor) -> tuple[torch.Tensor,]:
        """k-space (..., Z, Y, X) -> Image (..., Z, Y, X)."""
        # Multiply with Conj Phasor: (..., Z, Y, X) * (L, X) -> (L, ..., Z, Y, X)
        k_expanded = torch.einsum('...zyx, lx -> l...zyx', kspace, self.phasor.conj())

        # IFFT applied to each segment
        (img_seg,) = self._fft_op.adjoint(k_expanded)

        # Multiply with Masks and sum over segments: (L, ..., Z, Y, X) * (L, ..., Z, Y, X) -> (..., Z, Y, X)
        res = torch.einsum('l...zyx, l...zyx -> ...zyx', img_seg, self.masks.conj())

        return (res,)
