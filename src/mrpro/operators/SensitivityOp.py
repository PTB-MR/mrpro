"""Class for Sensitivity Operator."""

import torch

from mrpro.data.CsmData import CsmData
from mrpro.operators.LinearOperator import LinearOperator


class SensitivityOp(LinearOperator):
    """Sensitivity operator class.

    The forward operator expands an image to multiple coil images according to coil sensitivity maps,
    the adjoint operator reduces the coil images to a single image.
    """

    def __init__(self, csm: CsmData | torch.Tensor) -> None:
        """Initialize a Sensitivity Operator.

        Parameters
        ----------
        csm
           Coil Sensitivity Data
        """
        super().__init__()
        if isinstance(csm, CsmData):
            # only tensors can be used as buffers
            csm = csm.data
        self.csm_tensor = csm

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the forward operator, thus expand the coils dimension.

        Parameters
        ----------
        x
            image data tensor with dimensions `(other 1 z y x)`.

        Returns
        -------
            image data tensor with dimensions `(other coils z y x)`.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply SensitivityOp.

        Use `operator.__call__`, i.e. call `operator()` instead.
        """
        return (self.csm_tensor * x,)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint operator, thus reduce the coils dimension.

        Parameters
        ----------
        y
            image data tensor with dimensions `(other coils z y x)`.

        Returns
        -------
            image data tensor with dimensions `(other 1 z y x)`.
        """
        return ((self.csm_tensor.conj() * y).sum(-4, keepdim=True),)
