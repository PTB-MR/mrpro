"""K-space trajectory base class."""

from abc import ABC, abstractmethod

import torch
from einops import repeat

from mr2.data.KTrajectory import KTrajectory
from mr2.data.SpatialDimension import SpatialDimension


class KTrajectoryCalculator(ABC):
    """Base class for k-space trajectories."""

    @abstractmethod
    def __call__(
        self,
        *,
        n_k0: int,
        k0_center: int | torch.Tensor,
        k1_idx: torch.Tensor,
        k1_center: int | torch.Tensor,
        k2_idx: torch.Tensor,
        k2_center: int | torch.Tensor,
        encoding_matrix: SpatialDimension,
        reversed_readout_mask: torch.Tensor | None = None,
    ) -> KTrajectory:
        """Calculate the trajectory for given KHeader.

        The shapes of kz, ky and kx of the calculated trajectory must be
        broadcastable to (prod(all_other_dimensions), k2, k1, k0).

        Not all of the parameters will be used by all implementations.

        Parameters
        ----------
        n_k0
            number of samples in k0
        k0_center
            position of k-space center in k0
        k1_idx
            indices of k1
        k1_center
            position of k-space center in k1
        k2_idx
            indices of k2
        k2_center
            position of k-space center in k2
        reversed_readout_mask
            boolean tensor indicating reversed readout
        encoding_matrix
            encoding matrix

        Returns
        -------
            Trajectory

        """

    def _readout(
        self, n_k0: int, k0_center: int | torch.Tensor, reversed_readout_mask: torch.Tensor | None
    ) -> torch.Tensor:
        """Calculate the trajectory along one readout (k0 dimension).

        Parameters
        ----------
        n_k0
            number of samples in readout
        k0_center
            index of k-space center in readout
        reversed_readout_mask
            boolean tensor indicating reversed readout, e.g bipolar readout

        Returns
        -------
            trajectory along one readout

        """
        if isinstance(k0_center, int):
            k0_center = repeat(
                torch.tensor(k0_center), '... -> ... other coils k2 k1 k0', other=1, coils=1, k2=1, k1=1, k0=1
            )
        elif k0_center.ndim < 5:
            raise ValueError(f'Expected k0_center to have at least 5 dimensions, got {k0_center.ndim}.')
        k0 = torch.arange(n_k0, dtype=torch.float32) - k0_center
        if reversed_readout_mask is not None:
            # The -1 should not be required as far as we understand ISMRMRD, but
            # at least our Siemens epi data needs it.
            # See https://github.com/fzimmermann89/mr2/pull/832 for more details.
            k0 = torch.where(reversed_readout_mask.unsqueeze(-1), -1 - k0, k0)
        return k0

    def __repr__(self) -> str:
        """Get string representation."""
        return f'{type(self).__name__}()'


class DummyTrajectory(KTrajectoryCalculator):
    """Simple Dummy trajectory that returns zeros.

    Shape will fit to all data. Only used as dummy for testing.
    """

    def __call__(self, n_k0: int, k1_idx: torch.Tensor, k2_idx: torch.Tensor, **_) -> KTrajectory:
        """Calculate dummy trajectory."""
        shape = torch.broadcast_shapes(k1_idx.shape, k2_idx.shape)
        kx = torch.arange(shape.numel()).reshape(shape)
        ky = torch.zeros(*shape)
        kz = torch.arange(n_k0).reshape(1, 1, 1, n_k0)
        return KTrajectory(kz, ky, kx)
