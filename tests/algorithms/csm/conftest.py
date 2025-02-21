"""PyTest fixtures for the csm tests."""

import torch
from mrpro.data import IData, SpatialDimension
from mrpro.phantoms.coils import birdcage_2d


def multi_coil_image(n_coils, ph_ellipse, random_kheader, n_other=(1,)):
    """Create multi-coil image."""
    image_dimensions = SpatialDimension(z=1, y=ph_ellipse.n_y, x=ph_ellipse.n_x)

    # Create reference coil sensitivities
    csm_ref = birdcage_2d(n_coils, image_dimensions)

    # Create multi-coil phantom image data
    img = ph_ellipse.phantom.image_space(image_dimensions)
    # +1 to ensure that there is signal everywhere, for voxel == 0 csm cannot be determined.
    img_multi_coil = (img + 1) * csm_ref

    # Repeat data for multiple other dimensions
    img_multi_coil = torch.tile(img_multi_coil, n_other + (1,) * 4)
    idata = IData.from_tensor_and_kheader(data=img_multi_coil, kheader=random_kheader)
    return (idata, csm_ref)
