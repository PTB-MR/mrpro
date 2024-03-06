"""Numerical coil simulations."""

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

import numpy as np
import torch
from einops import repeat

from mrpro.data import SpatialDimension


def birdcage_2d(
    number_of_coils: int,
    image_dimensions: SpatialDimension[int],
    relative_radius: float = 1.5,
    normalize_with_rss: bool = True,
) -> torch.Tensor:
    """Numerical simulation of 2D Birdcage coil sensitivities.

    Parameters
    ----------
    number_of_coils
        number of coil elements
    image_dimensions
        number of voxels in the image
        This is a 2D simulation so the output will be (1 number_of_coils 1 image_dimensions.y image_dimensions.x)
    relative_radius
        relative radius of birdcage
    normalize_with_rss
        If set to true, the calculated sensitivities are normalized by the root-sum-of-squares

    This function is strongly inspired by https://github.com/ismrmrd/ismrmrd-python-tools. The associated license
    information can be found at the end of this file.
    """
    dim = [number_of_coils, image_dimensions.y, image_dimensions.x]
    x_co, y_co = torch.meshgrid(
        torch.linspace(-dim[2] // 2, dim[2] // 2 - 1, dim[2]),
        torch.linspace(-dim[1] // 2, dim[1] // 2 - 1, dim[1]),
        indexing='xy',
    )

    c = torch.linspace(0, dim[0] - 1, dim[0])[:, None, None]
    coil_center_x = dim[2] * relative_radius * np.cos(c * (2 * torch.pi / dim[0]))
    coil_center_y = dim[1] * relative_radius * np.sin(c * (2 * torch.pi / dim[0]))
    coil_phase = -c * (2 * torch.pi / dim[0])

    rr = torch.sqrt((x_co[None, ...] - coil_center_x) ** 2 + (y_co[None, ...] - coil_center_y) ** 2)
    phi = torch.arctan2((x_co[None, ...] - coil_center_x), -(y_co[None, ...] - coil_center_y)) + coil_phase
    sensitivities = (1 / rr) * np.exp(1j * phi)

    if normalize_with_rss:
        rss = torch.sqrt(torch.sum(torch.abs(sensitivities) ** 2, 0))
        # Normalize only where rss is > 0
        sensitivities[:, rss > 0] /= rss[None, rss > 0]

    return repeat(sensitivities, 'coils y x->other coils z y x', other=1, z=1)


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
