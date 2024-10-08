"""(Iterative) Walsh method for coil sensitivity map calculation."""

import torch

from mrpro.data.SpatialDimension import SpatialDimension
from mrpro.utils.filters import uniform_filter


def walsh(coil_images: torch.Tensor, smoothing_width: SpatialDimension[int] | int) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using an iterative version of the Walsh method.

    This is for a single set of coil images. The input should be a tensor with dimensions
    (coils, z, y, x). The output will have the same dimensions.
    Either apply this function individually to each set of coil images,
    or see CsmData.from_idata_walsh which performs this operation on a whole dataset [WAL2000]_.

    This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

    Parameters
    ----------
    coil_images
        images for each coil element
    smoothing_width
        width of the smoothing filter

    References
    ----------
    .. [WAL2000] Walsh DO, Gmitro AF, Marcellin MW (2000) Adaptive reconstruction of phased array MR imagery. MRM 43
    """
    # After 10 power iterations we will have a very good estimate of the singular vector
    n_power_iterations = 10

    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)
    # Compute the pointwise covariance between coils
    coil_covariance = torch.einsum('azyx,bzyx->abzyx', coil_images, coil_images.conj())

    # Smooth the covariance along y-x for 2D and z-y-x for 3D data
    coil_covariance = uniform_filter(coil_covariance, width=smoothing_width.zyx, dim=(-3, -2, -1))

    # At each point in the image, find the dominant eigenvector
    # of the signal covariance matrix using the power method
    v = coil_covariance.sum(dim=0)
    for _ in range(n_power_iterations):
        v /= v.norm(dim=0)
        v = torch.einsum('abzyx,bzyx->azyx', coil_covariance, v)
    csm = v / v.norm(dim=0)

    # Make sure there are no inf or nan-values due to very small values in the covariance matrix
    # nan_to_num does not work for complexfloat, boolean indexing not with vmap.
    csm = torch.where(torch.isfinite(csm), csm, 0.0)
    return csm
