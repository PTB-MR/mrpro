"""Walsh method for coil sensitivity map calculation."""

import torch

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.filters import uniform_filter


def walsh(
    coil_images: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
    align_phase: bool = False,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using Walsh's method [WAL2000]_.

    This function computes CSMs from a set of complex coil images assuming spatially
    slowly changing sensitivity maps using Walsh's method [WAL2000]_.

    The algorithm follows these steps:

    1. **Compute Pointwise Covariance**:
       Calculate the covariance matrix of the coil images at each voxel to capture inter-coil signal relationships.

    2. **Apply Smoothing Filter**:
       Smooth the covariance matrices across spatial dimensions using a uniform filter of specified width
       to reduce noise and enforce spatial consistency.

    3. **Dominant Eigenvector Estimation via Power Iteration**:
       Perform power iterations to approximate the dominant eigenvector of the covariance matrix at each voxel,
       representing the principal component of the signal.

    4. **Normalize Sensitivity Maps**:
       Normalize the resulting eigenvectors to produce the final CSMs.

    5. **Phase Alignment (Optional)**:
       If `align_phase` is True, aligns the eigenvectors' global phase to a reference derived from the
       coil data [INA2013]_. This prevents phase singularities that otherwise cause destructive
       interference when spatially interpolating or downsampling the maps.

    This function works on a single set of coil images. The input should be a tensor with dimensions
    `(... coils, z, y, x)`. The output will have the same dimensions. Either apply this function individually to
    each set of coil images, or see `~mrpro.data.CsmData.from_idata_walsh` which performs this operation on
    a whole dataset.

    Parameters
    ----------
    coil_images
        images for each coil element, shape (..., coils, z, y, x).
    smoothing_width
        width of the smoothing filter.
    align_phase
        if True, resolve the phase ambiguity of eigenvectors relative to the data [INA2013]_.

    Returns
    -------
    csm
        coil sensitivity map, shape (..., coils, z, y, x).

    References
    ----------
    .. [WAL2000] Walsh DO, Gmitro AF, Marcellin MW (2000) Adaptive reconstruction of phased array MR imagery. MRM 43
    .. [INA2013] Inati S, Hansen M, Kellman P (2013) A solution to the phase problem in adaptive coil combination.
       in Proceedings of the 21st Annual Meeting of ISMRM, Salt Lake City, USA, 2672.
    """
    n_power_iterations = 10
    eps = 1e-12

    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(
            z=smoothing_width if coil_images.shape[-3] > 1 else 1, y=smoothing_width, x=smoothing_width
        )

    # Pointwise covariance
    coil_covariance = torch.einsum('... a z y x, ... b z y x -> ... a b z y x', coil_images, coil_images.conj())

    # Smooth covariance
    coil_covariance = uniform_filter(coil_covariance, width=smoothing_width.zyx, dim=(-3, -2, -1))

    # Power iterations for dominant eigenvector
    v = coil_covariance.sum(dim=-4)
    for _ in range(n_power_iterations):
        v = v / (v.norm(dim=-4, keepdim=True) + eps)
        v = torch.einsum('... a b z y x, ... b z y x -> ... a z y x', coil_covariance, v)

    csm = v / (v.norm(dim=-4, keepdim=True) + eps)

    if align_phase:
        # Resolve global phase ambiguity using a low-res data projection
        d_sum = torch.sum(coil_images, dim=(-3, -2, -1), keepdim=True)
        d_sum /= d_sum.norm(dim=-4, keepdim=True) + eps
        phase_map = torch.einsum('... c z y x, ... c z y x -> ... z y x', d_sum.conj(), csm).angle()
        csm = csm * torch.exp(-1j * phase_map).unsqueeze(-4)

    csm = torch.where(torch.isfinite(csm), csm, 0.0)
    return csm
