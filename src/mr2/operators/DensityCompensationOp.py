"""Class for Density Compensation Operator."""

import torch

from mr2.data.DcfData import DcfData
from mr2.operators.EinsumOp import EinsumOp


class DensityCompensationOp(EinsumOp):
    """Density Compensation Operator."""

    def __init__(self, dcf: DcfData | torch.Tensor) -> None:
        """Initialize a Density Compensation Operator.

        Parameters
        ----------
        dcf
           Density Compensation Data
        """
        if isinstance(dcf, DcfData):
            # only tensors can currently be used as buffers
            # thus, einsumop is initialized with the tensor data
            # TODO: change if einsumop can handle dataclasses
            dcf_tensor = dcf.data
        else:
            dcf_tensor = dcf
        super().__init__(dcf_tensor, '...,... -> ...')

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply density compensation to k-space data.

        This operator performs an element-wise multiplication of the input k-space data
        with the density compensation factors (DCF).

        Parameters
        ----------
        x
            Input k-space data.

        Returns
        -------
            Density compensated k-space data.
        """
        return super().__call__(x)
