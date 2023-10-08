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

import dataclasses

import torch

from mrpro.data import IData
from mrpro.data import SpatialDimension
from mrpro.utils.filters import spatial_uniform_filter_3d


@dataclasses.dataclass(slots=True, frozen=True)
class CsmData(IData):
    """Coil sensitivity map class."""

    @staticmethod
    def _iterative_walsh_csm(
        coil_images: torch.Tensor, smoothing_width: SpatialDimension[int], niter: int
    ) -> torch.Tensor:
        """Calculate csm using an iterative version of the Walsh method.

        This function is strongly inspired by https://github.com/ismrmrd/ismrmrd-python-tools. The associated license
        information can be found at the end of this file.

        More information on the method can be found in
        https://doi.org/10.1002/(SICI)1522-2594(200005)43:5<682::AID-MRM10>3.0.CO;2-G

        Parameters
        ----------
        coil_images
            Images for each coil element.
        smoothing_width
            Width smoothing filter.
        niter
            Number if iterations of Walsh method.
        """
        ncoils, nz, ny, nx = coil_images.shape

        # Compute the sample covariance pointwise
        coil_cov = torch.zeros((ncoils, ncoils, nz, ny, nx), dtype=coil_images.dtype)
        for c1_idx in range(ncoils):
            for c2_idx in range(c1_idx):
                coil_cov[c1_idx, c2_idx, ...] = coil_images[c1_idx, ...] * torch.conj(coil_images[c2_idx, ...])
                coil_cov[c2_idx, c1_idx, ...] = torch.conj(coil_cov[c1_idx, c2_idx, ...].clone())
            coil_cov[c1_idx, c1_idx, ...] = coil_images[c1_idx, ...] * torch.conj(coil_images[c1_idx, ...])

        # Smooth the covariance along y-x for 2D and z-y-x for 3D data
        if coil_images.shape[-3] == 1:
            # 2D case
            smoothing_width = SpatialDimension(1, smoothing_width.y, smoothing_width.x)
        coil_cov = spatial_uniform_filter_3d(coil_cov, filter_width=smoothing_width)

        # At each point in the image, find the dominant eigenvector
        # and corresponding eigenvalue of the signal covariance
        # matrix using the power method
        csm_data = torch.zeros((ncoils, nz, ny, nx), dtype=coil_images.dtype)
        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    coil_cov_curr_vox = coil_cov[:, :, z, y, x]
                    # v needs to be (ncoils, 1) to allow for torch.mm later on
                    v = torch.sum(coil_cov_curr_vox, dim=0)[:, None]
                    lam = torch.linalg.norm(v)
                    if lam != 0:
                        v = v / lam

                        for _ in range(niter):
                            v = torch.mm(coil_cov_curr_vox, v)
                            lam = torch.linalg.norm(v)
                            v = v / lam

                    csm_data[:, z, y, x] = v[:, 0]

        # Make sure there are no inf and nan-values due to very small values in the covariance matrix
        csm_data[torch.isnan(csm_data) | torch.isinf(csm_data)] = 0
        return csm_data

    @classmethod
    def from_idata_walsh(
        cls,
        idata: IData,
        downsampling_factor: SpatialDimension[int] = SpatialDimension(1, 1, 1),
        smoothing_width: SpatialDimension[int] = SpatialDimension(5, 5, 5),
        niter: int = 3,
    ) -> CsmData:
        """Create csm object from image data using iterative Walsh method.

        Parameters
        ----------
        idata
            IData object containing the images for each coil element.
        smoothing_width
            Width of smoothing filter.
        niter
            Number if iterations of Walsh method.
        """
        csm_data = torch.zeros_like(idata.data)
        for ond in range(csm_data.shape[0]):
            csm_data[ond, ...] = CsmData._iterative_walsh_csm(idata.data[ond, ...], smoothing_width, niter)

        return cls(header=idata.header, data=csm_data)


# License information from https://github.com/ismrmrd/ismrmrd-python-tools

# ISMRMRD-PYTHON-TOOLS SOFTWARE LICENSE JULY 2016

# PERMISSION IS HEREBY GRANTED, FREE OF CHARGE, TO ANY PERSON OBTAINING
# A COPY OF THIS SOFTWARE AND ASSOCIATED DOCUMENTATION FILES (THE
# "SOFTWARE"), TO DEAL IN THE SOFTWARE WITHOUT RESTRICTION, INCLUDING
# WITHOUT LIMITATION THE RIGHTS TO USE, COPY, MODIFY, MERGE, PUBLISH,
# DISTRIBUTE, SUBLICENSE, AND/OR SELL COPIES OF THE SOFTWARE, AND TO
# PERMIT PERSONS TO WHOM THE SOFTWARE IS FURNISHED TO DO SO, SUBJECT TO
# THE FOLLOWING CONDITIONS:

# THE ABOVE COPYRIGHT NOTICE, THIS PERMISSION NOTICE, AND THE LIMITATION
# OF LIABILITY BELOW SHALL BE INCLUDED IN ALL COPIES OR REDISTRIBUTIONS
# OF SUBSTANTIAL PORTIONS OF THE SOFTWARE.

# SOFTWARE IS BEING DEVELOPED IN PART AT THE NATIONAL HEART, LUNG, AND BLOOD
# INSTITUTE, NATIONAL INSTITUTES OF HEALTH BY AN EMPLOYEE OF THE FEDERAL
# GOVERNMENT IN THE COURSE OF HIS OFFICIAL DUTIES. PURSUANT TO TITLE 17,
# SECTION 105 OF THE UNITED STATES CODE, THIS SOFTWARE IS NOT SUBJECT TO
# COPYRIGHT PROTECTION AND IS IN THE PUBLIC DOMAIN. EXCEPT AS CONTAINED IN
# THIS NOTICE, THE NAME OF THE AUTHORS, THE NATIONAL HEART, LUNG, AND BLOOD
# INSTITUTE (NHLBI), OR THE NATIONAL INSTITUTES OF HEALTH (NIH) MAY NOT
# BE USED TO ENDORSE OR PROMOTE PRODUCTS DERIVED FROM THIS SOFTWARE WITHOUT
# SPECIFIC PRIOR WRITTEN PERMISSION FROM THE NHLBI OR THE NIH.THE SOFTWARE IS
# PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
