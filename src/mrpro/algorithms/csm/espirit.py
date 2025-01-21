"""ESPIRIT method for coil sensitivity map calculation."""

import torch
from einops import rearrange

from mrpro.data.SpatialDimension import SpatialDimension


def espirit(
    coil_k_space: torch.Tensor,
    img_shape: SpatialDimension[int],
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop: float = 0.95,
    max_iter: int = 10,
) -> torch.Tensor:
    """Calculate a coil sensitivity map (csm) using the ESPIRIT method.

    This is for a single set of coil images. The input should be a tensor with dimensions (coils, z, y, x).
    The output will have the same dimensions. Either apply this function individually to each set of coil images, or see
    CsmData.from_idata_espirit which performs this operation on a whole dataset [UEC2013]_.

    This function inspired by https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/app.html#EspiritCalib.

    Parameters
    ----------
    coil_k_space
        k-space for each coil element
    img_shape
        shape of the image
    thresh, optional
        threshold for the values, by default 0.02
    kernel_width, optional
        edge size of the scanning window, by default 6
    crop, optional
        not implemented, by default 0.95
    max_iter, optional
        maxiumum amount of iteration, by default 10

    Returns
    -------
        csm of the shape (coils, z, y, x)

    References
    ----------
    .. [UEC2013] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M (2013) ESPIRiT
    - an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA. MRM 71:990–1001​
    """
    # expecting coil_k_space to be of shape (coils, z, y, x)
    # inspired by https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/app.html#EspiritCalib

    # Get calibration matrix.
    # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
    mat = coil_k_space
    for ax in (1, 2, 3):
        mat = mat.unfold(dimension=ax, size=min(coil_k_space.shape[ax], kernel_width), step=1)

    num_coils, _, _, _, c, b, a = mat.shape
    mat = rearrange(mat, 'coils z y x c b a -> (z y x) (coils c b a)')

    # Perform SVD on calibration matrix
    _, s, vh = torch.linalg.svd(mat, full_matrices=False)

    # Get kernels
    vh = torch.diag((s > thresh * s.max()).type(vh.type())) @ vh
    kernels = rearrange(vh, 'n (coils c b a) -> n coils c b a', coils=num_coils, c=c, b=b, a=a)

    # Get covariance matrix in image domain
    aha = torch.zeros((num_coils, num_coils, *img_shape), dtype=coil_k_space.dtype, device=coil_k_space.device)

    for kernel in kernels:
        img_kernel = torch.fft.ifftn(kernel, s=img_shape, dim=(-3, -2, -1))
        img_kernel = torch.fft.ifftshift(img_kernel, dim=(-1, -2, -3))
        aha += torch.einsum('c z y x, d z y x->c d z y x ', img_kernel, img_kernel.conj())

    aha *= aha[0, 0].numel() / kernels.shape[-1]

    v = aha.sum(dim=0)
    for _ in range(max_iter):
        v /= v.norm(dim=0)
        v = torch.einsum('abzyx,bzyx->azyx', aha, v)
    max_eig = v.norm(dim=0)
    csm = v / max_eig

    # Normalize phase with respect to first channel
    csm *= csm[0].conj() / csm[0].abs()

    # Crop maps by thresholding eigenvalue
    csm *= max_eig  # > crop

    return csm
