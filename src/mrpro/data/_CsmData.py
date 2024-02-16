"""Class for coil sensitivity maps (csm)."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations
from einops import rearrange

import torch

from mrpro.data import IData
from mrpro.data import QData
from mrpro.data import KData

from mrpro.data import SpatialDimension
from mrpro.utils.filters import spatial_uniform_filter_3d


class CsmData(QData):
    """Coil sensitivity map class."""

    @staticmethod
    def _iterative_walsh_csm(
        coil_images: torch.Tensor, smoothing_width: SpatialDimension[int], niter: int
    ) -> torch.Tensor:
        """Calculate csm using an iterative version of the Walsh method.

        This function is inspired by https://github.com/ismrmrd/ismrmrd-python-tools.

        More information on the method can be found in
        https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

        Parameters
        ----------
        coil_images
            images for each coil element.
        smoothing_width
            width smoothing filter.
        niter
            number of iterations of Walsh method.
        """
        # Compute the pointwise covariance between coils
        coil_cov = torch.einsum('azyx,bzyx->abzyx', coil_images, coil_images.conj())

        # Smooth the covariance along y-x for 2D and z-y-x for 3D data
        coil_cov = spatial_uniform_filter_3d(coil_cov, filter_width=smoothing_width)

        # At each point in the image, find the dominant eigenvector
        # of the signal covariance matrix using the power method
        v = coil_cov.sum(dim=0)
        for _ in range(niter):
            v /= v.norm(dim=0)
            v = torch.einsum('abzyx,bzyx->azyx', coil_cov, v)
        csm_data = v / v.norm(dim=0)

        # Make sure there are no inf or nan-values due to very small values in the covariance matrix
        # nan_to_num does not work for complexfloat, boolean indexing not with vmap.
        csm_data = torch.where(torch.isfinite(csm_data), csm_data, 0.0)
        return csm_data

    @classmethod
    def estimate_walsh(
        cls,
        idata: IData,
        smoothing_width: SpatialDimension[int] = SpatialDimension(5, 5, 5),
        niter: int = 3,
        chunk_size_otherdim: int | None = None,
    ) -> CsmData:
        """Create csm object from image data using iterative Walsh method.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            width of smoothing filter.
        niter
            number of iterations of Walsh method.
        chunk_size_otherdim:
            How many elements of the other dimensions should be processed at once.
            Default is None, which means that all elements are processed at once.
        """
        csm_fun = torch.vmap(
            lambda img: CsmData._iterative_walsh_csm(img, smoothing_width, niter),
            chunk_size=chunk_size_otherdim,
        )
        csm_data = csm_fun(idata.data)

        return cls(header=idata.header, data=csm_data)

    def estimate_espirit(
        cls,
        kdata: KData,
        thresh: float = 0.02,
        kernel_width: int = 6,
        max_iter: int = 10,
        crop: float = 0.95,
        chunk_size_otherdim=None,
    ) -> CsmData:
        """Espirit sensitivity Estimation (DRAFT)

        Works only for Cartesian K Data

        Parameters
        ----------
        kdata
            _description_
        chunk_size_otherdim, optional
            _description_, by default None

        """
        # check for certesian
        # get calib
        calib = kdata.data
        img_shape = kdata.data.shape[-3:]

        csm_fun = torch.vmap(
            lambda c: CsmData._espirit_csm(
                c, img_shape=img_shape, thresh=thresh, kernel_width=kernel_width, max_iter=max_iter, crop=crop
            ),
            chunk_size=chunk_size_otherdim,
        )
        csm_data = csm_fun(calib)
        return cls(header=kdata.header, data=csm_data)

    def _espirit_csm(
        calib: torch.Tensor,
        img_shape,
        thresh=0.02,
        kernel_width=6,
        crop=0.95,
        max_iter=10,
    ):
        # inspired by https://sigpy.readthedocs.io/en/latest/_modules/sigpy/mri/app.html#EspiritCalib

        # Get calibration matrix.
        # Shape [num_coils] + num_blks + [kernel_width] * img_ndim
        mat = calib
        for ax in (1, 2, 3):
            mat = mat.unfold(dimension=ax, size=min(calib.shape[ax], kernel_width), step=1)
        mat = rearrange(mat, 'coils z y x c b a -> (z y x) (coils c b a)')

        # Perform SVD on calibration matrix
        _, S, VH = torch.linalg.svd(mat, full_matrices=False)

        # Get kernels
        kernels = rearrange(VH[S > thresh * S.max()], 'n (coils cba) -> n coils cba')

        # Get covariance matrix in image domain
        num_coils = calib.shape[1]
        AHA = torch.zeros(num_coils, num_coils, img_shape, dtype=calib.dtype, device=calib.device)

        for kernel in kernels:
            img_kernel = torch.fft.ifftn(kernel, s=img_shape, dim=(-1, -2, -3))
            AHA += torch.einsum('c z y x, d z y x->c d z y x ', img_kernel, img_kernel.conj())

        AHA *= AHA[0, 0].numel() / kernels.shape[-1]

        v = AHA.sum(dim=0)
        for _ in range(max_iter):
            v /= v.norm(dim=0)
            v = torch.einsum('abzyx,bzyx->azyx', AHA, v)
        max_eig = v.norm(dim=0)
        csm = v / max_eig

        # Normalize phase with respect to first channel
        csm *= csm[0].conj() / csm[0].abs()

        # Crop maps by thresholding eigenvalue
        csm *= max_eig > crop

        return csm
