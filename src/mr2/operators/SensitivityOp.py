"""Class for Sensitivity Operator."""

import torch

from mr2.data.CsmData import CsmData
from mr2.operators.LinearOperator import LinearOperator


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

    def __call__(self, img: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply coil sensitivity maps to expand the coil dimension.

        This operator multiplies the input image (assumed to be a single coil image
        or a combined image) by the coil sensitivity maps (CSM) to produce
        multi-coil image data.

        Parameters
        ----------
        img
            Input image data, typically with shape `(... 1 z y x)`.

        Returns
        -------
            Multi-coil image data with shape `(... coils z y x)`.
        """
        return super().__call__(img)

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply forward of SensitivityOp.

        .. note::
            Prefer calling the instance of the SensitivityOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return (self.csm_tensor * img,)

    def adjoint(self, img: torch.Tensor) -> tuple[torch.Tensor,]:
        """Apply adjoint sensitivity mapping to reduce the coil dimension.

        This operator multiplies the input multi-coil image data by the complex
        conjugate of the coil sensitivity maps (CSM) and then sums along
        the coil dimension, effectively performing a coil combination.

        Parameters
        ----------
        img
            Multi-coil image data, typically with shape `(... coils z y x)`.

        Returns
        -------
            Combined image data, with shape `(... 1 z y x)`.
        """
        return ((self.csm_tensor.conj() * img).sum(-4, keepdim=True),)
