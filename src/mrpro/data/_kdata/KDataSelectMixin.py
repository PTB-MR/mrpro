"""Select subset along other dimensions of KData."""

import copy
from typing import Literal

import torch
from typing_extensions import Self

from mrpro.data._kdata.KDataProtocol import _KDataProtocol
from mrpro.data.Rotation import Rotation


class KDataSelectMixin(_KDataProtocol):
    """Select subset of KData."""

    def select_other_subset(
        self: Self,
        subset_idx: torch.Tensor,
        subset_label: Literal['average', 'slice', 'contrast', 'phase', 'repetition', 'set'],
    ) -> Self:
        """Select a subset from the other dimension of KData.

        Parameters
        ----------
        kdata
            K-space data (other coils k2 k1 k0)
        subset_idx
            Index which elements of the other subset to use, e.g. phase 0,1,2 and 5
        subset_label
            Name of the other label, e.g. phase

        Returns
        -------
            K-space data (other_subset coils k2 k1 k0)

        Raises
        ------
        ValueError
            If the subset indices are not available in the data
        """
        # Make a copy such that the original kdata.header remains the same
        kheader = copy.deepcopy(self.header)
        ktraj = self.traj.as_tensor()

        # Verify that the subset_idx is available
        label_idx = getattr(kheader.acq_info.idx, subset_label)
        if not all(el in torch.unique(label_idx) for el in subset_idx):
            raise ValueError('Subset indices are outside of the available index range')

        # Find subset index in acq_info index
        other_idx = torch.cat([torch.where(idx == label_idx[:, 0, 0])[0] for idx in subset_idx], dim=0)

        # Adapt header
        kheader.acq_info.apply_(
            lambda field: field[other_idx, ...] if isinstance(field, torch.Tensor | Rotation) else field
        )

        # Select data
        kdat = self.data[other_idx, ...]

        # Select ktraj
        if ktraj.shape[1] > 1:
            ktraj = ktraj[:, other_idx, ...]

        return type(self)(kheader, kdat, type(self.traj).from_tensor(ktraj))
