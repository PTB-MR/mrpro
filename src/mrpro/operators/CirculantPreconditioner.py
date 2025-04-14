"""A preconditioner for non-Cartesian iterative SENSE reconstruction."""

import torch

from mrpro.data.DcfData import DcfData
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.NonUniformFastFourierOp import NonUniformFastFourierOp


class CirculantPreconditioner(LinearOperator):
    """A preconditioner for a non-Cartesian SENSE reconstruction."""

    def __init__(self, nufft_operator: NonUniformFastFourierOp, dcf: DcfData):
        """Initialize a circulant preconditioner for a non-Cartesian SENSE reconstruction.

        This operator acts as a preconditioner P for iterative algorithms
        solving Ax=b, where A involves NUFFT operations (F) andoptionally
        coil sensitivities (C), e.g., A = C^H F^H F C.

        The preconditioner approximates the inverse of the density-compensated
        operator, P ≈ (F^H W F)^(-1), where W represents the density
        compensation factors (DCF). It is constructed by:
        1. Simulating the density-compensated Point Spread Function (PSF)
        h_w = F^H W F(delta).
        2. Computing the FFT of the PSF. These are the
        eigenvalues of the circulant approximation.
        3. Regularizing and inverting these eigenvalues to get the k-space
        kernel

        This preconditioner is suitable for accelerating solvers for the
        *unweighted* least-squares problem (where A does not include W),
        as it compensates for density variations internally.

        Parameters
        ----------
        nufft_operator
            The non-uniform fast fourier transform operator.
        dcf
            density compensation weights
        """
        super().__init__()
        delta_image = torch.zeros(nufft_operator._im_size, dtype=torch.complex64)
        center_indices = tuple(d // 2 for d in nufft_operator._im_size)
        delta_image[center_indices] = 1.0
        delta_image = delta_image.to(dcf.device)
        (k,) = nufft_operator(delta_image)
        k = k * dcf.data
        (psf,) = nufft_operator.adjoint(k)
        kernel = torch.fft.fftn(psf, dim=(-1, -2, -3))
        kernel = torch.polar(kernel.abs().clamp(min=1e-5), kernel.angle()).reciprocal()
        self.kernel = kernel

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the inverse of the preconditioner."""
        x = torch.fft.fftn(x, dim=(-1, -2, -3))
        x = x * self.kernel
        x = torch.fft.ifftn(x, dim=(-1, -2, -3))
        return (x,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint of the inverse of the preconditioner."""
        y = torch.fft.fftn(y, dim=(-1, -2, -3))
        y = y * self.kernel.conj()
        y = torch.fft.ifftn(y, dim=(-1, -2, -3))
        return (y,)
