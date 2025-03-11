"""2D radial analytical density compensation function."""

import torch

from mrpro.algorithms.dcf import dcf_1d


def extract_angle_distance_along_spoke_unique(traj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the constant angle and unique distance along spokes using Singular Value Decomposition (SVD).

    This function computes the principal spoke direction via SVD and projects each k-space
    (kx, ky) point onto this direction to obtain its signed distance from the origin.
    It ensures that the computed distances are identical across acquisitions and collapses
    them into a single 1D vector.

    Parameters
    ----------
    traj : torch.Tensor
        k-space trajectory positions, shaped `[2, 1, acquisitions, spokes]`.

    Returns
    -------
    angles : torch.Tensor
        The computed angles for each acquisition, shaped `[acquisitions]`.

    distances_along_spoke : torch.Tensor
        The unique signed distances along the spoke, shaped `[spokes]`.

    Raises
    ------
    ValueError
        If the computed distances are not unique across acquisitions.
    """
    m = traj.squeeze(1).permute(1, 2, 0)
    _, _, v = torch.svd(m)
    principal_directions = v[:, :, 0]

    angles = torch.atan2(principal_directions[:, 1], principal_directions[:, 0])
    angles %= torch.pi

    distances_along_spoke = torch.einsum('aji,ai->aj', m, principal_directions)

    # Ensure all acquisitions have the same distances
    if not torch.allclose(distances_along_spoke[0], distances_along_spoke):
        raise ValueError('Distances along spokes are not unique across acquisitions!')

    distances_along_spoke_unique = distances_along_spoke[0]

    return angles, distances_along_spoke_unique


def dcf_2dradial(traj: torch.Tensor) -> torch.Tensor:
    """Calculate sample density compensation function for an 2D radial trajectory.

    The density compensation function is calculated by calculating the fraction of the area of the k-space.
    The dcf is calculated along the polar angle and the spoke dim and is then combined.
    The calculation is based on the formula from
    https://users.fmrib.ox.ac.uk/~karla/reading_group/lecture_notes/AdvRecon_Pauly_read.pdf for equidistant radial
    sampling.

    Parameters
    ----------
    traj
        k-space positions `(2, 1, k1, k0)`

    Returns
    -------
        density compensation values for analytical radial trajectory `(1, 1, k1, k0)`
    """
    angles, distances_along_spoke_unique = extract_angle_distance_along_spoke_unique(traj)

    # get dcf along the polar angle dim which should sum to 1 to get the fraction of the areas
    dcf_polar = dcf_1d(angles, periodicity=torch.pi)
    dcf_polar /= dcf_polar.sum()

    # get dcf along the spoke dim, this is mainly pi*r^2 of the outer ring - pi*r^2 of the inner ring
    dcf_spoke = torch.pi * dcf_1d(distances_along_spoke_unique**2)

    dcf = torch.outer(dcf_polar, dcf_spoke)

    # fix center value
    zero_indices = torch.nonzero(distances_along_spoke_unique == 0)

    # Assert that there is only one zero in the array
    if len(zero_indices) != 1:
        raise ValueError('The array should contain exactly one zero.')

    center_idx = int(zero_indices[0].item())

    dcf[:, center_idx] = dcf_spoke[center_idx] / 4 / len(angles)

    return dcf.unsqueeze(0).unsqueeze(0)
