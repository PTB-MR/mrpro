"""Sampling from von Mises-Fisher distribution."""

# based on: https://github.com/jasonlaska/spherecluster/blob/701b0b1909088a56e353b363b2672580d4fe9d93/spherecluster/util.py
# http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python
# https://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
# http://www.stat.pitt.edu/sungkyu/software/randvonMisesFisher3.pdf

from math import log, sqrt

import torch


def sample_vmf(mu: torch.Tensor, kappa: float, n_samples: int) -> torch.Tensor:
    """
    Generate samples from the von Mises-Fisher distribution.

    The von Mises-Fisher distribution is a circular normal distribution on the unit hypersphere.

    Parameters
    ----------
    mu
        Center of the distribution on the unit hypersphere. Shape: (..., dim)
    kappa
        Concentration parameter.
        For small kappa, the distribution is close to uniform.
        For large kappa, the distribution is close to a normal distribution with variance 1/kappa.
    n_samples
        Number of samples to generate.

    Returns
    -------
        Samples from the von Mises-Fisher distribution. Shape: (num_samples, ..., dim)
    """
    mu_ = mu.unsqueeze(0) if mu.dim() == 1 else mu
    total_samples = n_samples * mu_[..., 0].numel()
    mu_ = mu_.expand((n_samples, *mu_.shape))
    dim = mu_.shape[-1]

    b = (dim - 1) / (sqrt(4.0 * kappa**2 + (dim - 1) ** 2) + 2 * kappa)
    x = (1.0 - b) / (1.0 + b)
    c = kappa * x + (dim - 1) * log(1 - x**2)

    beta_dist = torch.distributions.Beta((dim - 1) / 2.0, (dim - 1) / 2.0)
    uniform_dist = torch.distributions.Uniform(0, 1)
    normal_dist = torch.distributions.Normal(0, 1)

    ws: list[torch.Tensor] = []

    while sum(len(w) for w in ws) < total_samples:
        # rejection sampling
        z = beta_dist.sample(torch.Size((total_samples,)))
        w = (1.0 - (1.0 + b) * z) / (1.0 - (1.0 - b) * z)
        u = uniform_dist.sample(torch.Size((total_samples,)))
        accepted = kappa * w + (dim - 1) * torch.log(1.0 - x * w) - c >= torch.log(u)
        ws.append(w[accepted])
    weights = torch.cat(ws)[:total_samples].reshape(mu_.shape[:-1])

    v = normal_dist.sample(mu_.shape)
    orthogonal_vectors = v - (mu_ * v).sum(-1, keepdim=True) * mu_ / mu_.norm(dim=-1, keepdim=True)
    orthonormal_vectors = orthogonal_vectors / orthogonal_vectors.norm(dim=-1, keepdim=True)
    samples = orthonormal_vectors * (1.0 - weights**2).sqrt().unsqueeze(-1) + weights.unsqueeze(-1) * mu_
    if mu.dim() == 1:
        samples = samples.squeeze(-2)
    return samples
