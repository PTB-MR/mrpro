"""Numerical coil simulations."""

import torch
from einops import repeat

from mrpro.data.SpatialDimension import SpatialDimension


def birdcage_2d(
    number_of_coils: int,
    image_dimensions: SpatialDimension[int],
    relative_radius: float = 1.5,
    normalize_with_rss: bool = True,
) -> torch.Tensor:
    """Numerical simulation of 2D Birdcage coil sensitivities.

    This function is strongly inspired by ISMRMRD Python Tools [ISMc]_. The associated license
    information can be found at the end of this file.

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

    References
    ----------
    .. [ISMc] ISMRMRD Python tools https://github.com/ismrmrd/ismrmrd-python-tools
    """
    dim = [number_of_coils, image_dimensions.y, image_dimensions.x]
    x_co, y_co = torch.meshgrid(
        torch.linspace(-dim[2] // 2, dim[2] // 2 - 1, dim[2]),
        torch.linspace(-dim[1] // 2, dim[1] // 2 - 1, dim[1]),
        indexing='xy',
    )

    x_co = repeat(x_co, 'y x -> coils y x', coils=1)
    y_co = repeat(y_co, 'y x -> coils y x', coils=1)

    c = repeat(torch.linspace(0, dim[0] - 1, dim[0]), 'coils -> coils y x', y=1, x=1)
    coil_center_x = dim[2] * relative_radius * torch.cos(c * (2 * torch.pi / dim[0]))
    coil_center_y = dim[1] * relative_radius * torch.sin(c * (2 * torch.pi / dim[0]))
    coil_phase = -c * (2 * torch.pi / dim[0])

    rr = torch.sqrt((x_co - coil_center_x) ** 2 + (y_co - coil_center_y) ** 2)
    phi = torch.arctan2((x_co - coil_center_x), -(y_co - coil_center_y)) + coil_phase
    sensitivities = (1 / rr) * torch.exp(1j * phi)

    if normalize_with_rss:
        rss = sensitivities.abs().square().sum(0).sqrt()
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
