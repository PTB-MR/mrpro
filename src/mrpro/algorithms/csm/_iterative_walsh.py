import torch

from mrpro.data._SpatialDimension import SpatialDimension
from mrpro.utils.filters import uniform_filter_3d


def iterative_walsh(
    coil_images: torch.Tensor,
    smoothing_width: SpatialDimension[int] | int,
    power_iterations: int,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using an iterative version of the Walsh method.

    This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

    More information on the method can be found in
    https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

    Parameters
    ----------
    coil_images
        images for each coil element
    smoothing_width
        width of the smoothing filter
    power_iterations
        number of iterations used to determine dominant eigenvector
    """
    if isinstance(smoothing_width, int):
        smoothing_width = SpatialDimension(smoothing_width, smoothing_width, smoothing_width)
    # Compute the pointwise covariance between coils
    coil_cov = torch.einsum('azyx,bzyx->abzyx', coil_images, coil_images.conj())

    # Smooth the covariance along y-x for 2D and z-y-x for 3D data
    coil_cov = uniform_filter_3d(coil_cov, filter_width=smoothing_width)

    # At each point in the image, find the dominant eigenvector
    # of the signal covariance matrix using the power method
    v = coil_cov.sum(dim=0)
    for _ in range(power_iterations):
        v /= v.norm(dim=0)
        v = torch.einsum('abzyx,bzyx->azyx', coil_cov, v)
    csm_data = v / v.norm(dim=0)

    # Make sure there are no inf or nan-values due to very small values in the covariance matrix
    # nan_to_num does not work for complexfloat, boolean indexing not with vmap.
    csm_data = torch.where(torch.isfinite(csm_data), csm_data, 0.0)
    return csm_data
