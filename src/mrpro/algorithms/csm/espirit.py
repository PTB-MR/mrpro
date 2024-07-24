"""ESPIRIT method for coil sensitivity map calculation."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from einops import rearrange


def espirit(
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
    num_coils, _, _, _, c, b, a = mat.shape
    mat = rearrange(mat, 'coils z y x c b a -> (z y x) (coils c b a)')

    # Perform SVD on calibration matrix
    _, S, VH = torch.linalg.svd(mat, full_matrices=False)

    # Get kernels
    VH = torch.diag((S > thresh * S.max()).type(VH.type())) @ VH
    kernels = rearrange(VH, 'n (coils c b a) -> n coils c b a', coils=num_coils, c=c, b=b, a=a)

    # Get covariance matrix in image domain
    AHA = torch.zeros((num_coils, num_coils, *img_shape), dtype=calib.dtype, device=calib.device)

    for kernel in kernels:
        img_kernel = torch.fft.ifftn(kernel, s=img_shape, dim=(-3, -2, -1))
        img_kernel = torch.fft.ifftshift(img_kernel, dim=(-1, -2, -3))
        AHA += torch.einsum('c z y x, d z y x->c d z y x ', img_kernel, img_kernel.conj())

    AHA *= AHA[0, 0].numel() / kernels.shape[-1]

    v = AHA.sum(dim=0)
    for _ in range(max_iter):
        v /= v.norm(dim=0)
        v = torch.einsum('abzyx,bzyx->azyx', AHA, v)
    max_eig = v.norm(dim=0)
    print(max_eig.max())
    csm = v / max_eig

    # Normalize phase with respect to first channel
    csm *= csm[0].conj() / csm[0].abs()

    # Crop maps by thresholding eigenvalue
    csm *= max_eig #> crop

    return csm
