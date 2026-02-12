"""Inati method for coil sensitivity map calculation."""

import torch
from einops import einsum

from mr2.data.SpatialDimension import SpatialDimension
from mr2.utils.sliding_window import sliding_window


def inati(
    coil_img: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using the Inati method [INA2013]_ [INA2014]_.

    This is for a single set of coil images. The input should be a tensor with dimensions `(coils, z, y, x)`. The output
    will have the same dimensions. Either apply this function individually to each set of coil images, or see
    `~mr2.data.CsmData.from_idata_inati` which performs this operation on a whole dataset.

    .. [INA2013] Inati S, Hansen M, Kellman P (2013) A solution to the phase problem in adaptvie coil combination.
       in Proceedings of the 21st Annual Meeting of ISMRM, Salt Lake City, USA, 2672.

    .. [INA2014] Inati S, Hansen M (2014) A Fast Optimal Method for Coil Sensitivity Estimation and Adaptive Coil
       Combination for Complex images. in Proceedings of Joint Annual Meeting ISMRM-ESMRMB, Milan, Italy, 7115.

    Parameters
    ----------
    coil_img
        images for each coil element
    smoothing_width
        Size of the smoothing kernel
    """
    # After 10 power iterations we will have a very good estimate of the singular vector
    n_power_iterations = 10
    eps = 1e-8  # for numerical stability

    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(
            z=smoothing_width if coil_img.shape[-3] > 1 else 1, y=smoothing_width, x=smoothing_width
        )

    if any(ks % 2 != 1 for ks in [smoothing_width.z, smoothing_width.y, smoothing_width.x]):
        raise ValueError('kernel_size must be odd')

    ks_halved = [ks // 2 for ks in smoothing_width.zyx]
    padded_coil_img = torch.nn.functional.pad(
        coil_img,
        (ks_halved[-1], ks_halved[-1], ks_halved[-2], ks_halved[-2], ks_halved[-3], ks_halved[-3]),
        mode='replicate',
    )
    # Get the voxels in an ROI defined by the smoothing_width around each voxel leading to shape
    # (z y x coils window=prod(smoothing_width))
    coil_img_roi = sliding_window(padded_coil_img, smoothing_width.zyx, dim=(-3, -2, -1)).flatten(-3)
    coil_img_cov = einsum(
        coil_img_roi.conj(),
        coil_img_roi,
        '... coils1 window,... coils2 window->... coils1 coils2',
    )

    singular_vector = torch.sum(coil_img_roi, dim=-1)  # z y x coils
    singular_vector /= singular_vector.norm(dim=-1, keepdim=True) + eps
    for _ in range(n_power_iterations):
        singular_vector = einsum(coil_img_cov, singular_vector, '... coils1 coils2,... coils2->... coils1')
        singular_vector /= singular_vector.norm(dim=-1, keepdim=True) + eps

    singular_value = einsum(coil_img_roi, singular_vector, '... coils window,... coils->... window')
    phase = singular_value.sum(-1)
    phase /= phase.abs() + eps
    csm = einsum(singular_vector.conj(), phase, '... coils,...->coils ...')  # coils z y x
    return csm
