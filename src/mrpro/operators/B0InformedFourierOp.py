"""B0-Informed Fourier Operator."""

import torch

from mrpro.operators.FastFourierOp import FastFourierOp
from mrpro.operators.LinearOperator import LinearOperator


class B0InformedFourierOp(LinearOperator):
    """B0-informed (fast) Fourier operator class.

    This operator implements a B0-informed Fourier transform that accounts for off-resonance effects.
    It supports different modes of operation dependent on the input parameters:

    .. note:
        If `num_time_points` and `num_frequencies` are both `None` or `0`,
        exact conjugate phase reconstruction with all individual frequencies is performed.
        If only `num_time_points` is set, time-segmented approximation is used for efficient computation.
        If only `num_frequencies` is set, conjugate phase reconstruction with frequency binning is performed.
        If both `num_time_points` and `num_frequencies` are set, time-segmented approximation with frequency
        binning is applied.
    """

    def __init__(
        self,
        b0_map: torch.Tensor,
        readout_times: torch.Tensor,
        fft_op: FastFourierOp,
        num_time_points: int | None = None,
        num_frequencies: int | None = None,
        b0_decimals: int = 0,
    ) -> None:
        """Initialize B0 informed Fourier operator.

        Parameters
        ----------
        b0_map
            Off-resonance map in Hz. Shape (..., Z, Y, X).
        readout_times
            Readout time vector in seconds. Shape (X,).
        fft_op
            Underlying Fast Fourier Transform operator.
        num_time_points
            Number of time segments for approximation.
        num_frequencies
            Number of frequency bins.
        b0_decimals
            Number of decimals to round B0 map values to after converting to radians/sec.
        """
        super().__init__()

        # Validate inputs
        if readout_times.shape[-1] != b0_map.shape[-1]:
            raise ValueError('readout_times must have same size as last dim of b0_map')

        device = b0_map.device
        self._fft_op: FastFourierOp = fft_op.to(device)
        b0_map_rad = torch.round(2 * torch.pi * b0_map, decimals=b0_decimals)  # Convert to radians/sec

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
                frequencies_rad = torch.quantile(b0_map_rad.flatten(), q_steps)
                weights = None
            else:
                frequencies_rad, counts = torch.unique(b0_map_rad, return_counts=True)
                weights = counts.to(dtype=b0_map_rad.dtype).sqrt()

            # Solve for Interpolator C: Solve the system A @ C = B in a least-squares sense.
            # A (Segment Phase): exp(-i * w * t_seg)
            # B (Readout Phase): exp(-i * w * t_ro)

            # Broadcasting: w (N_freq, 1) * t (1, N_time) -> (N_freq, N_time)
            # Segmented phase values from segmented time vector with shape (N_freqs, L)
            phase_segments = torch.exp(-1j * frequencies_rad[:, None] * t_seg[None, :])
            # Target phase values from readout times with shape (N_freqs, N_ro)
            phase_readout = torch.exp(-1j * frequencies_rad[:, None] * readout_times[None, :])

            # Apply weighting to the system if using Unique mode
            if weights is not None:
                phase_segments *= weights[:, None]
                phase_readout *= weights[:, None]

            # Interpolator obtained by pinv(phase_segments) @ phase_readout with shape (L, N_ro)
            temporal_basis = torch.linalg.lstsq(phase_segments, phase_readout, rcond=1e-15).solution

            # Precompute spatial basis functions B_l(r) = exp(-i * w(r) * t_seg_l)
            # with shape (L, ..., Z, Y, X) where ... are the spatial dimensions of b0_map.
            spatial_basis = torch.exp(-1j * torch.einsum('l, ... -> l...', t_seg, b0_map_rad))

        # Mode 2: Exact CPR with or without Frequency Binning
        # Uses spatial masks (basis) and exact temporal evolution (phasor)
        else:
            # Determine Frequencies and Spatial Indices
            if num_frequencies is not None and num_frequencies > 0:
                # Quantile Binning
                q_steps = torch.linspace(0, 1, num_frequencies + 1, device=device)
                boundaries = torch.quantile(b0_map_rad.flatten(), q_steps)

                # Use bin centers for frequency evolution
                frequencies_rad = (boundaries[:-1] + boundaries[1:]) / 2

                # Quantize/bucketize the map, indices will range [0, num_frequencies-1]
                indices = torch.bucketize(b0_map_rad, boundaries[:-1], right=True) - 1
                indices.clamp_(0, num_frequencies - 1)
            else:
                # Exact Unique Frequencies
                frequencies_rad, indices = torch.unique(b0_map_rad, return_inverse=True)
                indices = indices.view(b0_map_rad.shape)

            num_bins = len(frequencies_rad)

            # Compute Phasor (Temporal Evolution) with shape: (N_bins, N_readout)
            temporal_basis = torch.exp(-1j * torch.einsum('l, x -> lx', frequencies_rad, readout_times))

            # Compute Basis (One-Hot Masks) with shape: (number of frequencies/bins, *b0map shape)
            # WARNING: This can be memory intensive if number of bins, i.e. the number of frequencies is large
            idx_range = torch.arange(num_bins, device=device).view(-1, *([1] * b0_map_rad.ndim))
            spatial_basis = (indices.unsqueeze(0) == idx_range).to(dtype=torch.complex64)

        # Register buffers
        self._spatial_basis: torch.Tensor
        self._temporal_basis: torch.Tensor
        self.register_buffer('_spatial_basis', spatial_basis)  # Time-Seg -> Complex Maps | Binning -> Boolean Masks
        self.register_buffer('_temporal_basis', temporal_basis)  # Time-Seg -> Interpolator | Binning -> Time Evolution

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
        x_l = torch.einsum('...zyx, l...zyx -> l...zyx', x, self._spatial_basis)

        # Apply Fourier transform to each segment
        (k_l,) = self._fft_op(x_l)

        # Multiply with phasor and sum over segments
        res = torch.einsum('l...zyx, lx -> ...zyx', k_l, self._temporal_basis)

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
        y_l = torch.einsum('...zyx, lx -> l...zyx', y, self._temporal_basis.conj())

        # Inverse Fourier operator applied to each segment
        (x_l,) = self._fft_op.adjoint(y_l)

        # Multiply with Masks and sum over segments
        res = torch.einsum('l...zyx, l...zyx -> ...zyx', x_l, self._spatial_basis.conj())

        return (res,)
