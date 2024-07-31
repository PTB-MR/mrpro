"""Class for Density Compensation Operator."""

import torch

from mrpro.data.DcfData import DcfData
from mrpro.operators.EinsumOp import EinsumOp


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
        super().__init__(dcf_tensor, '... k2 k1 k0 ,... coil k2 k1 k0 ->... coil k2 k1 k0')
