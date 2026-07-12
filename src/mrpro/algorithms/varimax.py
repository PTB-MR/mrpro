"""Varimax rotation."""

import torch
from einops import rearrange


def varimax(phi: torch.Tensor, gamma: float = 1.0, n_iterations: int = 20) -> torch.Tensor:
    """Apply an orthogonal Varimax rotation.

    Parameters
    ----------
    phi
        Input matrix of shape `(*batch, n_components, n_channels)`.
        Rotation is performed across the `n_components` dimension.
    gamma
        Power parameter for the orthomax criterion. `1.0` gives Varimax.
    n_iterations
        Number of rotation iterations.

    Returns
    -------
        Rotated matrix with the same shape as `phi`.
    """
    if n_iterations < 0:
        raise ValueError('Number of iterations must be non-negative.')

    loadings = rearrange(phi, '... components channels -> ... channels components')
    n_components = loadings.shape[-1]
    rotation = torch.eye(n_components, dtype=phi.dtype, device=phi.device)

    for _ in range(n_iterations):
        rotated = loadings @ rotation
        squared_magnitude = rotated.abs().square()
        mean_squared_magnitude = squared_magnitude.mean(dim=-2, keepdim=True)
        weighted_loadings = rotated * (squared_magnitude - gamma * mean_squared_magnitude)
        u, _, vh = torch.linalg.svd(loadings.mH @ weighted_loadings)
        rotation = u @ vh

    return rearrange(loadings @ rotation, '... channels components -> ... components channels')
