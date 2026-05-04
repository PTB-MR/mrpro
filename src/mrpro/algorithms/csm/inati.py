"""Inati method for coil sensitivity map calculation."""

import torch
from einops import einsum

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.filters import uniform_filter


def inati(
    coil_img: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
    n_iterations: int = 5,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using the iterative Inati method [INA2014]_.

    This function computes CSMs using an iterative alternating minimization approach that estimates
    the combined structural image and the coil sensitivity profiles simultaneously. Compared to
    covariance-based methods, this approach is more memory-efficient and inherently enforces
    spatial phase coherence.

    The algorithm follows these steps:

    1. **Initialize Combined Image**:
       Create an initial coil combination using a global sum of the coil data.

    2. **Iterative Refinement**:
       For a specified number of iterations:
       a. Update the CSMs by dividing the coil images by the combined image and smoothing the results.
       b. Update the combined image using the newly estimated CSMs (SENSE combination).
       c. Align the global phase at each step to avoid phase singularities and ensure spatial smoothness.

    3. **Final Normalization**:
       Normalize the sensitivity maps to have unit 2-norm across the coil dimension.

    Parameters
    ----------
    coil_img
        images for each coil element, shape (..., coils, z, y, x).
    smoothing_width
        size of the smoothing kernel.
    n_iterations
        number of iterations to refine the maps. Default is 5.

    Returns
    -------
    csm
        coil sensitivity map, shape (..., coils, z, y, x).

    References
    ----------
    .. [INA2013] Inati S, Hansen M, Kellman P (2013) A solution to the phase problem in adaptive coil combination.
       in Proceedings of the 21st Annual Meeting of ISMRM, Salt Lake City, USA, 2672.
    .. [INA2014] Inati S, Hansen M, Kellman P (2014) A Fast Optimal Method for Coil Sensitivity Estimation and
       Adaptive Coil Combination for Complex Images. in Proceedings of Joint Annual Meeting ISMRM-ESMRMB, Milan,
       Italy, 7115.
    """
    eps = 1e-12

    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(
            z=smoothing_width if coil_img.shape[-3] > 1 else 1,
            y=smoothing_width,
            x=smoothing_width,
        )

    if any(ks % 2 != 1 for ks in [smoothing_width.z, smoothing_width.y, smoothing_width.x]):
        raise ValueError('kernel_size must be odd')

    # Initial guess for combined image phase
    d_sum = torch.sum(coil_img, dim=(-3, -2, -1))
    d_sum /= d_sum.norm(dim=-1, keepdim=True) + eps
    combined_img = einsum(d_sum.conj(), coil_img, '... c, ... c z y x -> ... z y x')

    for _ in range(n_iterations):
        # Update CSM
        csm = coil_img * combined_img.conj().unsqueeze(-4)
        csm = uniform_filter(csm, width=smoothing_width.zyx, dim=(-3, -2, -1))
        csm /= csm.norm(dim=-4, keepdim=True) + eps

        # Update Combined Image
        combined_img = einsum(coil_img, csm.conj(), '... c z y x, ... c z y x -> ... z y x')

        # Compute global phase reference and align
        d_sum = (csm * combined_img.unsqueeze(-4)).sum(dim=(-3, -2, -1))
        d_sum /= d_sum.norm(dim=-1, keepdim=True) + eps

        phase = einsum(d_sum.conj(), csm, '... c, ... c z y x -> ... z y x').angle()
        combined_img *= torch.exp(1j * phase)
        csm *= torch.exp(-1j * phase).unsqueeze(-4)

    return csm
