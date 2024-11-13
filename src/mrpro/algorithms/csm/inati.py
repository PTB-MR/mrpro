"""Inati method for coil sensitivity map calculation."""

import torch

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.sliding_window import sliding_window


def inati(
    coil_images: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using an the Inati method [INA2013]_ [INA2014]_.

    This is for a single set of coil images. The input should be a tensor with dimensions (coils, z, y, x). The output
    will have the same dimensions. Either apply this function individually to each set of coil images, or see
    CsmData.from_idata_inati which performs this operation on a whole dataset.

    .. [INA2013] Inati S, Hansen M, Kellman P (2013) A solution to the phase problem in adaptvie coil combination.
       in Proceedings of the 21st Annual Meeting of ISMRM, Salt Lake City, USA, 2672.

    .. [INA2014] Inati S, Hansen M (2014) A Fast Optimal Method for Coil Sensitivity Estimation and Adaptive Coil
       Combination for Complex Images. in Proceedings of Joint Annual Meeting ISMRM-ESMRMB, Milan, Italy, 7115.

    Parameters
    ----------
    coil_images
        Images for each coil element
    smoothing_width
        Size of the smoothing kernel
    """
    # After 10 power iterations we will have a very good estimate of the singular vector
    n_power_iterations = 10

    # Padding at the edge of the images
    padding_mode = 'replicate'

    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(
            z=smoothing_width if coil_images.shape[-3] > 1 else 1, y=smoothing_width, x=smoothing_width
        )

    if any(ks % 2 != 1 for ks in [smoothing_width.z, smoothing_width.y, smoothing_width.x]):
        raise ValueError('kernel_size must be odd')

    ks_halved = [ks // 2 for ks in smoothing_width.zyx]
    padded_coil_images = torch.nn.functional.pad(
        coil_images,
        (ks_halved[-1], ks_halved[-1], ks_halved[-2], ks_halved[-2], ks_halved[-3], ks_halved[-3]),
        mode=padding_mode,
    )
    # Get the voxels in an ROI defined by the smoothing_width around each voxel leading to shape
    # (coils z y x prod(smoothing_width))
    coil_images_roi = sliding_window(padded_coil_images, smoothing_width.zyx, axis=(-3, -2, -1)).flatten(-3)
    # Covariance with shape (z y x coils coils)
    coil_images_covariance = torch.einsum('i...j,k...j->...ik', coil_images_roi.conj(), coil_images_roi)
    singular_vector = torch.sum(coil_images_roi, dim=-1)  # coils z y x
    singular_vector /= singular_vector.norm(dim=0, keepdim=True)
    for _ in range(n_power_iterations):
        singular_vector = torch.einsum('...ij,j...->i...', coil_images_covariance, singular_vector)  # coils z y x
        singular_vector /= singular_vector.norm(dim=0, keepdim=True)

    singular_value = torch.einsum('i...j,i...->...j', coil_images_roi, singular_vector)  # z y x prod(smoothing_width)
    phase = singular_value.sum(-1)
    phase /= phase.abs()  # z y x
    csm = singular_vector.conj() * phase[None, ...]
    return csm
