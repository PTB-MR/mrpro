"""Conjugate Phase Fast Fourier Transform operator."""

import torch

from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator


class B0InformedFourierOp(LinearOperator):
    """Optimized 3D B0 Informed Reconstruction Operator."""

    _basis: torch.Tensor
    _phasor: torch.Tensor
    _fft_op: FastFourierOp

    def __init__(
        self,
        b0_map: torch.Tensor,
        readout_times: torch.Tensor,
        fft_op: FastFourierOp,
        num_time_points: int | None = None,
        num_frequencies: int | None = None,
    ) -> None:
        """Initialize Conjugate Phase Operator.

        Args:
            b0_map (torch.Tensor): Off-resonance map in Hz. Shape (..., Z, Y, X).
            readout_times (torch.Tensor): Readout time vector in seconds. Shape (X,).
            num_time_points (int | None): Number of time segments for approximation.
            num_frequencies (int | None): Number of frequency bins.
        """
        super().__init__()

        # Validate inputs
        if readout_times.shape[-1] != b0_map.shape[-1]:
            raise ValueError('readout_times must have same size as last dim of b0_map')

        device = b0_map.device
        ndim = b0_map.ndim
        self._fft_op = fft_op.to(device)

        # Mode 1: Time-Segmented Approximation
        # Solves for an interpolator (C) such that: Basis(t_seg) @ C ≈ Basis(t_readout)
        if num_time_points is not None and num_time_points > 0:
            # Define Time Segments
            t_seg = torch.linspace(readout_times[0], readout_times[-1], num_time_points, device=device)

            # Determine Frequency Basis for Least Squares Design
            # If num_frequencies is set, we use Quantiles to design the filter.
            # Otherwise, we use all Unique Frequencies (weighted by their occurrence count).
            if num_frequencies is not None and num_frequencies > 0:
                q_steps = torch.linspace(0, 1, num_frequencies, device=device)
                frequencies = torch.quantile(b0_map.flatten(), q_steps)
                weights = None
            else:
                frequencies, counts = torch.unique(b0_map, return_counts=True)
                weights = counts.to(dtype=b0_map.dtype).sqrt()

            # Solve for Interpolator C: Solve the system A @ C = B in a least-squares sense.
            # A (Segment Phase): exp(-i * w * t_seg)
            # B (Readout Phase): exp(-i * w * t_ro)

            # Broadcasting: w (N_freq, 1) * t (1, N_time) -> (N_freq, N_time)
            # Segmented phase values from segmented time vector with shape (N_freqs, L)
            phase_segments = torch.exp(-2j * torch.pi * frequencies[:, None] * t_seg[None, :])
            # Target phase values from readout times with shape (N_freqs, N_ro)
            phase_readout = torch.exp(-2j * torch.pi * frequencies[:, None] * readout_times[None, :])

            # Apply weighting to the system if using Unique mode
            if weights is not None:
                phase_segments *= weights[:, None]
                phase_readout *= weights[:, None]

            # Interpolator obtained by pinv(phase_segments) @ phase_readout with shape (L, N_ro)
            phasor = torch.linalg.lstsq(phase_segments, phase_readout, rcond=1e-15).solution

            # Precompute spatial basis functions B_l(r) = exp(-i * w(r) * t_seg_l)
            # with shape (L, ..., Z, Y, X) where ... are the spatial dimensions of b0_map.
            basis = torch.exp(-1j * 2 * torch.pi * torch.einsum('l, ... -> l...', t_seg, b0_map))

        # Mode 2: Exact CPR with or without Frequency Binning
        # Uses spatial masks (basis) and exact temporal evolution (phasor)
        else:
            # Determine Frequencies and Spatial Indices
            if num_frequencies is not None and num_frequencies > 0:
                # Quantile Binning
                q_steps = torch.linspace(0, 1, num_frequencies + 1, device=device)
                boundaries = torch.quantile(b0_map.flatten(), q_steps)

                # Use bin centers for frequency evolution
                frequencies = (boundaries[:-1] + boundaries[1:]) / 2

                # Digitizing the map (Bucketize)
                # indices will range [0, num_frequencies-1]
                indices = torch.bucketize(b0_map, boundaries[:-1], right=True) - 1
                indices.clamp_(0, num_frequencies - 1)
            else:
                # Exact Unique Frequencies
                frequencies, indices = torch.unique(b0_map, return_inverse=True)
                indices = indices.view(b0_map.shape)

            frequencies = 2 * torch.pi * frequencies
            num_bins = len(frequencies)

            # Compute Phasor (Temporal Evolution) with shape: (N_bins, N_readout)
            phasor = torch.exp(-1j * frequencies[:, None] * readout_times[None, :])

            # Compute Basis (One-Hot Masks) with shape: (N_bins, Z, Y, X)
            # WARNING: This can be memory intensive if num_bins is large (Exact mode)
            idx_range = torch.arange(num_bins, device=device).view(-1, *([1] * ndim))
            basis = (indices.unsqueeze(0) == idx_range).to(dtype=torch.complex64)

        # Register buffers
        self.register_buffer('_basis', basis)  # Time-Seg -> Complex Maps | Binning -> Boolean Masks
        self.register_buffer('_phasor', phasor)  # Time-Seg -> Interpolator | Binning -> Time Evolution

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply B0-informed Fourier transform to input tensor (image to k-space).

        This method multiplies the input with basis functions, applies FT to each segment,
        and combines results using a phasor weighting scheme.

        Parameters
        ----------
        x
            Input image tensor of shape (..., z, y, x) to be transformed.

        Returns
        -------
            Tuple containing the transformed k-space tensor of shape (..., z, y, x).
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of FourierOp.

        .. note::
            Prefer calling the instance of the FourierOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        # Multiply with masks / basis functions
        x_l = torch.einsum('...zyx, l...zyx -> l...zyx', x, self._basis)

        # Apply Fourier transform to each segment
        (k_l,) = self._fft_op(x_l)

        # Multiply with phasor and sum over segments
        res = torch.einsum('l...zyx, lx -> ...zyx', k_l, self._phasor)

        return (res,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the B0-informed fast Fourier operator.

        Computes the adjoint operation by applying the conjugate phasor multiplication,
        inverse FFT transformation, and basis function weighting to recover the original
        signal space representation.

        Parameters
        ----------
        y
            Input k-spacetensor in Fourier space with shape (..., z, y, x).

        Returns
        -------
            Reconstructed image tensor in signal space with shape (..., z, y, x).
        """
        # Multiply with conjugate phasor
        y_l = torch.einsum('...zyx, lx -> l...zyx', y, self._phasor.conj())

        # Inverse Fourier operator applied to each segment
        (x_l,) = self._fft_op.adjoint(y_l)

        # Multiply with Masks and sum over segments
        res = torch.einsum('l...zyx, l...zyx -> ...zyx', x_l, self._basis.conj())

        return (res,)
