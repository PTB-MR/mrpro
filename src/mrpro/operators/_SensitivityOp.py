"""Class for Sensitivity Operator."""

import torch

from mrpro.data import CsmData


class SensitivityOp:  # todo: later put (LinearOp) here
    """Sensitivity operator class."""

    def __init__(self, csm: CsmData):
        self.C = csm.data

    def forward(self, u: torch.Tensor):
        """Apply the forward operator, thus expand the num_coils dimension."""
        Cu = self.C * u
        return Cu

    def adjoint(self, v: torch.Tensor):
        """Apply the adjoint operator, thus reduce the num_coils dimension."""
        C_H = torch.conj(self.C)
        C_Hv = torch.sum(C_H * v, dim=1).unsqueeze(1)
        return C_Hv
