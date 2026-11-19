"""Dictionary Operator."""

from collections.abc import Sequence

import torch

from mrpro.operators.AnalysisDictionaryOp import AnalysisDictionaryOp
from mrpro.operators.LinearOperator import LinearOperator


class SynthesisDictionaryOp(LinearOperator):
    r"""Synthesis dictionary operator (coefficients -> patches).

    Mathematically, this operator implements the multiplication of a synthesis dictionary (i.e. a collection of
    :math:`K` sparsifying functions, so-called atoms) with a :math:`K`-dimensional coefficient vector, i.e.
    :math:`D s`, where :math:`D \in V^{d \times K}` with :math:`V \in \{\mathbb{R}, \mathbb{C}\}` is the synthesis
    dictionary (typically with :math:`K \geq d`) and :math:`s \in V^K` is the coefficient vector.

    This operator is intended for use with image patches. Therefore, the dictionary :math:`D` is defined by its
    :math:`K` sparsifying functions of shape :math:`d_1 \times d_2 \times \ldots`, with
    :math:`d := d_1 \cdot d_2 \cdot \ldots`. The operator performs the matrix-vector multiplication described above
    with respect to the dimensions specified by the parameter `dim`. To achieve this, both the dictionary and the
    input are reshaped accordingly.

    For example, for a dictionary of shape ``(48, 8, 8)``, a set of coefficient vectors of shape
    ``(1000, 2, 160, 48)`` and ``dim=(-2, -1)``, the operator yields an output of shape ``(1000, 2, 160, 8, 8)``,
    where the last two dimensions correspond to the dimensions of the patches.

    The adjoint (Hermitian) operator maps the patches to the coefficient vectors, i.e. it implements
    :math:`D^H x`.
    """

    def __init__(self, dictionary: torch.Tensor, dim: Sequence[int] | int) -> None:
        """Initialize Synthesis Dictionary Operator.

        Parameters
        ----------
        dictionary
            Dictionary tensor to be applied to the coefficients.
            Shape e.g. `(n_entries, d1, d0)` for 2D patches or for 3D patches `(n_entries, d3, d1, d0)`
        dim
            Dimensions of the patches. Their order has to match the order of the patch
            dimensions of `dictionary`.
        """
        super().__init__()
        self._analysis = AnalysisDictionaryOp(dictionary, dim)

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply dictionary synthesis operation to input coefficients.

        The selected dimensions of the input patches are rearranged and flattened before applying the dictionary.

        Parameters
        ----------
        x
            Input tensor containing the coefficients.

        Returns
        -------
            Patches
        """
        return super().__call__(x)

    def forward(self, c: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of SynthesisDictionaryOp.

        .. note::
            Prefer calling the instance of the SynthesisDictionaryOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self._analysis.adjoint(c)

    def adjoint(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply dictionary analysis operation to input patches.

        The selected dimensions of the input patches are rearranged and flattened before applying the dictionary.

        Parameters
        ----------
        x
            Input tensor containing the patches.

        Returns
        -------
            Coefficients.
        """
        return self._analysis(x)
