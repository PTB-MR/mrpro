"""B0-informed Fourier operators."""

from abc import ABC, abstractmethod

import einops
import torch

from mrpro.operators.LinearOperator import LinearOperator


class B0InformedFourierOp(LinearOperator, ABC):
    """Abstract base class for B0-informed Fourier operators.

    Accounts for off-resonance effects via a separable approximation of the
    phase accumulation term.
    Base class for Multi-Frequency Interpolation, Time-Segmented Reconstruction and
    Conjugate Phase Fourier operators.
    """

    def __init__(
        self,
        fourier_op: LinearOperator,
        b0_map: torch.Tensor,
        readout_times: torch.Tensor,
    ) -> None:
        """Initialize B0-informed Fourier Operator.

        Parameters
        ----------
        fourier_op
            Underlying Fourier operator.
        b0_map
            Off-resonance map in Hz. Shape (..., z, y, x).
        readout_times
            Readout time vector in seconds. Shape (samples,).
        """
        super().__init__()
        self._fourier_op = fourier_op
        b0_map_rad = 2 * torch.pi * b0_map
        self._spatial_basis, self._temporal_basis = self._compute_basis(b0_map_rad, readout_times)

    @abstractmethod
    def _compute_basis(
        self, b0_map_rad: torch.Tensor, readout_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the spatial and temporal bases."""
        ...

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply B0-informed Fourier transform to input tensor.

        This transforms image data to k-space data, accounting for off-resonance effects.

        Parameters
        ----------
        x
            Input image tensor with shape (..., coils, z, y, x).

        Returns
        -------
            Transformed k-space tensor with shape (..., coils, k2, k1, k0).
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of B0InformedFourierOp.

        .. note::
            Prefer calling the instance of the operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        img_weighted = einops.einsum(x, self._spatial_basis, '... z y x, l ... z y x -> l ... z y x')
        (k_weighted,) = self._fourier_op(img_weighted)
        k = einops.einsum(k_weighted, self._temporal_basis, 'l ... z y x, l x -> ... z y x')
        return (k,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the B0-informed Fourier operator.

        This transforms k-space data to image data, accounting for off-resonance effects.

        Parameters
        ----------
        y
            Input k-space tensor with shape (..., coils, k2, k1, k0).

        Returns
        -------
            Reconstructed image tensor in signal space with shape (..., coils, z, y, x).
        """
        k_weighted = einops.einsum(y, self._temporal_basis.conj(), '... z y x, l x -> l ... z y x')
        (img_weighted,) = self._fourier_op.adjoint(k_weighted)
        img = einops.einsum(
            img_weighted,
            self._spatial_basis.conj(),
            'l ... z y x, l ... z y x -> ... z y x',
        )
        return (img,)


class MultiFrequencyFourierOp(B0InformedFourierOp):
    """Multi-Frequency Interpolation (MFI) B0-informed Fourier operator.

    Approximates the off-resonance phase term by defining discrete frequency
    bins and using soft spatial interpolation to compute image components.

    References
    ----------
    .. [1] Man LC, Pauly JM, Macovski A. Multifrequency interpolation for fast
       off-resonance correction. Magn Reson Med. 1997;37(5):785-792.
    """

    def __init__(
        self,
        fourier_op: LinearOperator,
        b0_map: torch.Tensor,
        readout_times: torch.Tensor,
        n_bins: int = 32,
    ) -> None:
        """Initialize Multi-Frequency Fourier Operator.

        Parameters
        ----------
        fourier_op
            Underlying Fourier operator.
        b0_map
            Off-resonance map in Hz. Shape (..., z, y, x).
        readout_times
            Readout time vector in seconds. Shape (samples,).
        n_bins
            Number of frequency bins.
        """
        if n_bins <= 0:
            raise ValueError('n_bins must be strictly positive.')
        self.n_bins = n_bins
        super().__init__(fourier_op, b0_map, readout_times)

    def _compute_basis(
        self, b0_map_rad: torch.Tensor, readout_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            quantile_steps = torch.linspace(0, 1, self.n_bins, device=b0_map_rad.device, dtype=b0_map_rad.dtype)
            frequency_centers = torch.quantile(b0_map_rad.flatten(), quantile_steps)
            frequency_centers = torch.unique(frequency_centers)

        temporal_basis = torch.exp(-1j * frequency_centers[:, None] * readout_times[None, :]).to(
            b0_map_rad.dtype.to_complex()
        )

        if frequency_centers.numel() == 1:
            spatial_basis = torch.ones(
                (1, *b0_map_rad.shape), dtype=b0_map_rad.dtype.to_complex(), device=b0_map_rad.device
            )
            return spatial_basis, temporal_basis

        n_centers = frequency_centers.numel()
        b0_map_flat = b0_map_rad.reshape(-1)

        with torch.no_grad():
            idx_right = torch.bucketize(b0_map_flat, frequency_centers).clamp(1, n_centers - 1)
            idx_left = idx_right - 1
            left_freq = frequency_centers[idx_left]
            right_freq = frequency_centers[idx_right]
            delta_freq = (right_freq - left_freq).clamp_min(1e-12)

        spatial_basis = torch.zeros((n_centers, b0_map_flat.numel()), device=b0_map_rad.device, dtype=b0_map_rad.dtype)
        w = (b0_map_flat - left_freq) / delta_freq
        spatial_basis.scatter_add_(0, idx_left[None, :], (1 - w)[None, :])
        spatial_basis.scatter_add_(0, idx_right[None, :], w[None, :])
        spatial_basis = spatial_basis.reshape(n_centers, *b0_map_rad.shape).to(b0_map_rad.dtype.to_complex())

        return spatial_basis, temporal_basis


class TimeSegmentedFourierOp(B0InformedFourierOp):
    """Time-Segmented Reconstruction (TSR) B0-informed Fourier operator.

    Approximates the phase term by dividing the readout into time segments and
    using least-squares optimized interpolators for temporal components.

    References
    ----------
    .. [1] Noll DC, Meyer CH, Pauly JM, Nishimura DG, Macovski A. A homogeneity
       correction method for magnetic resonance imaging with time-varying
       gradients. IEEE Trans Med Imaging. 1991;10(4):629-637.
    """

    def __init__(
        self,
        fourier_op: LinearOperator,
        b0_map: torch.Tensor,
        readout_times: torch.Tensor,
        n_segments: int = 32,
        n_design_frequencies: int = 64,
    ) -> None:
        """Initialize Time-Segmented Fourier Operator.

        Parameters
        ----------
        fourier_op
            Underlying Fourier operator.
        b0_map
            Off-resonance map in Hz. Shape (..., z, y, x).
        readout_times
            Readout time vector in seconds. Shape (samples,).
        n_segments
            Number of time segments.
        n_design_frequencies
            Number of frequencies for least-squares design.
        """
        if n_segments <= 0:
            raise ValueError('n_segments must be strictly positive.')
        self.n_segments = n_segments
        self.n_design_frequencies = n_design_frequencies
        super().__init__(fourier_op, b0_map, readout_times)

    def _compute_basis(
        self, b0_map_rad: torch.Tensor, readout_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_segments = torch.linspace(readout_times[0], readout_times[-1], self.n_segments, device=b0_map_rad.device)

        with torch.no_grad():
            quantile_steps = torch.linspace(
                0, 1, self.n_design_frequencies, device=b0_map_rad.device, dtype=b0_map_rad.dtype
            )
            design_frequencies = torch.unique(torch.quantile(b0_map_rad.flatten(), quantile_steps))

            segment_phases = torch.exp(-1j * design_frequencies[:, None] * time_segments[None, :])
            target_phases = torch.exp(-1j * design_frequencies[:, None] * readout_times[None, :])
            temporal_basis = torch.linalg.lstsq(segment_phases, target_phases, rcond=1e-15).solution.to(
                b0_map_rad.dtype.to_complex()
            )

        spatial_basis = torch.exp(-1j * einops.einsum(time_segments, b0_map_rad, 'l, ... -> l ...')).to(
            b0_map_rad.dtype.to_complex()
        )

        return spatial_basis, temporal_basis


class ConjugatePhaseFourierOp(B0InformedFourierOp):
    """Conjugate Phase (CP) B0-informed Fourier operator.

    Performs an exact direct evaluation of the phase accumulation integral
    without separable approximation. Extremely computationally expensive.

    References
    ----------
    .. [1] Maeda A, Sano K, Yokoyama T. Reconstruction by two-dimensional
       time phase correction. IEEE Trans Med Imaging. 1988;7(1):26-31.
    """

    def _compute_basis(
        self, b0_map_rad: torch.Tensor, readout_times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        spatial_basis = torch.exp(-1j * einops.einsum(readout_times, b0_map_rad, 'l, ... -> l ...')).to(
            b0_map_rad.dtype.to_complex()
        )
        temporal_basis = torch.eye(readout_times.numel(), dtype=b0_map_rad.dtype.to_complex(), device=b0_map_rad.device)
        return spatial_basis, temporal_basis
