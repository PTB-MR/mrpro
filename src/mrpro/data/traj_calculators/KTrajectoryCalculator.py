"""K-space trajectory base class."""

from abc import ABC, abstractmethod

import torch

from mrpro.data.KTrajectory import KTrajectory
from mrpro.data.KTrajectoryRawShape import KTrajectoryRawShape
from mrpro.data.SpatialDimension import SpatialDimension


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
    ) -> KTrajectory | KTrajectoryRawShape:
        """Calculate the trajectory for given KHeader.

        The shapes of kz, ky and kx of the calculated trajectory must be
        broadcastable to (prod(all_other_dimensions), k2, k1, k0).

        Not all of the parameters will be used by all implementations.

        Parameters
        ----------
        n_k0
            number of samples in k0
        k1_idx
            indices of k1
        k2_idx
            indices of k2
        k0_center
            position of k-space center in k0
        k1_center
            position of k-space center in k1
        k2_center
            position of k-space center in k2
        reversed_readout_mask
            boolean tensor indicating reversed redout
        encoding_matrix
            encoding matrix, describing the extend of the k-space coordinates



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
            position of k-space center in readout
        reversed_readout_mask
            boolean tensor indicating reversed readout, e.g bipolar readout

        Returns
        -------
            trajectory along one readout

        """
        k0 = torch.linspace(0, n_k0 - 1, n_k0, dtype=torch.float32) - k0_center
        # Data can be obtained with standard or reversed readout (e.g. bipolar readout).
        if reversed_readout_mask is not None:
            k0, reversed_readout_mask = torch.broadcast_tensors(k0, reversed_readout_mask)
            k0[reversed_readout_mask] = torch.flip(k0[reversed_readout_mask], (-1,))
        return k0


class DummyTrajectory(KTrajectoryCalculator):
    """Simple Dummy trajectory that returns zeros.

    Shape will fit to all data. Only used as dummy for testing.
    """

    def __call__(self, **_) -> KTrajectory:
        """Calculate dummy trajectory."""
        kx = torch.zeros(1, 1, 1, 1)
        ky = torch.zeros(1, 1, 1, 1)
        kz = torch.zeros(1, 1, 1, 1)
        return KTrajectory(kz, ky, kx)
