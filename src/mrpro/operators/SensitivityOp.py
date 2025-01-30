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

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the forward operator, thus expand the coils dimension.

        Parameters
        ----------
        img
            image data tensor with dimensions `(other 1 z y x)`.

        Returns
        -------
            image data tensor with dimensions `(other coils z y x)`.
        """
        return (self.csm_tensor * img,)

    def adjoint(self, img: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply the adjoint operator, thus reduce the coils dimension.

        Parameters
        ----------
        img
            image data tensor with dimensions `(other coils z y x)`.

        Returns
        -------
            image data tensor with dimensions `(other 1 z y x)`.
        """
        return ((self.csm_tensor.conj() * img).sum(-4, keepdim=True),)
