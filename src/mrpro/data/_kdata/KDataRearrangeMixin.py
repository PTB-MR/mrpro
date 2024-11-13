"""Rearrange KData."""

import copy

from einops import rearrange
from typing_extensions import Self

from mrpro.data._kdata.KDataProtocol import _KDataProtocol
from mrpro.data.AcqInfo import rearrange_acq_info_fields


class KDataRearrangeMixin(_KDataProtocol):
    """Rearrange KData."""

    def rearrange_k2_k1_into_k1(self: Self) -> Self:
        """Rearrange kdata from (... k2 k1 ...) to (... 1 (k2 k1) ...).

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)

        Returns
        -------
            K-space data (other coils 1 (k2 k1) k0)
        """
        # Rearrange data
        kdat = rearrange(self.data, '... coils k2 k1 k0->... coils 1 (k2 k1) k0')

        # Rearrange trajectory
        ktraj = rearrange(self.traj.as_tensor(), 'dim ... k2 k1 k0-> dim ... 1 (k2 k1) k0')

        # Create new header with correct shape
        kheader = copy.deepcopy(self.header)

        # Update shape of acquisition info index
        kheader.acq_info.apply_(
            lambda field: rearrange_acq_info_fields(field, 'other k2 k1 ... -> other 1 (k2 k1) ...')
        )

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))
