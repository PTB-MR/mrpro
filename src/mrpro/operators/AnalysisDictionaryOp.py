"""Dictionary Operator."""

from collections.abc import Sequence

import torch

from mrpro.operators.EinsumOp import EinsumOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.RearrangeOp import RearrangeOp


def build_rearrange_and_flatten_pattern(dim: Sequence[int]) -> str:
    """Create string pattern to rearrange and flatten tensor.

    Build an einops rearrange pattern that moves the axes in `dim` to the end (in the given order), without needing
    to know the total number of dimensions. The specified `dim` are then flattened.

    Parameters
    ----------
    dim
        Axis indices (positive from the front and/or negative from the back) to move to the end, in the desired final
        order.

    Returns
    -------
    Einops pattern string, e.g. for dim=(1, -1, -2):
    'd0 d1 ... dm2 dm1 -> d0 ... (d1 dm1 dm2)'
    """
    front = [d for d in dim if d >= 0]
    back = [d for d in dim if d < 0]

    front_named = list(range(max(front) + 1)) if front else []
    back_named = list(range(min(back), 0)) if back else []

    def name(d: int) -> str:
        return f'd{d}' if d >= 0 else f'dm{-d}'

    left_parts = [name(d) for d in front_named] + ['...'] + [name(d) for d in back_named]

    unselected_front = [name(d) for d in front_named if d not in dim]
    unselected_back = [name(d) for d in back_named if d not in dim]
    selected = [name(d) for d in dim]

    right_parts = [*unselected_front, '...', *unselected_back, '(', *selected, ')']

    return f'{" ".join(left_parts)} -> {" ".join(right_parts)}'


class AnalysisDictionaryOp(LinearOperator):
    r"""Analysis dictionary operator (patches -> coefficients).

    Mathematically, this operator implements the multiplication of an analysis dictionary (i.e. a collection of
    :math:`K` sparsifying functions, so-called atoms) with a :math:`d`-dimensional signal, i.e. :math:`A x`, where
    :math:`A \in V^{K \times d}` with :math:`V \in \{\mathbb{R}, \mathbb{C}\}` is the analysis dictionary (typically
    with :math:`K \geq d`) and :math:`x \in V^d` is the signal. This operator is intended for use with image patches.
    Therefore, the operator :math:`A` is defined by its :math:`K` sparsifying functions of shape
    :math:`d_1 \times d_2 \times \ldots`, with :math:`d := d_1 \cdot d_2 \cdot \ldots`. This operator performs the
    just-described matrix-vector multiplication with respect to the dimensions specified by the parameter `dim`. To
    achieve this, both the dictionary and the input are reshaped accordingly.

    For example, for a dictionary of shape ``(48, 8, 8)``, an input corresponding to a set of patches of shape
    ``(1000, 2, 160, 8, 8)`` and ``dim=(-2, -1)``, the operator yields an output of shape ``(1000, 2, 160, 48)``,
    where the last dimension of size 48 contains the inner product of each atom (viewed as a vector) with the
    corresponding patch (viewed as a vector).

    The adjoint (Hermitian) operator maps the coefficient vectors to the patches, i.e. it implements :math:`A^H s`.
    """

    def __init__(self, dictionary: torch.Tensor, dim: Sequence[int] | int) -> None:
        """Initialize Analysis Dictionary Operator.

        Parameters
        ----------
        dictionary
            Dictionary tensor to be applied to the flattened patches.
            Shape e.g. `(n_entries, d1, d0)` for 2D patches or for 3D patches `(n_entries, d2, d1, d0)`
        dim
            Dimensions of the patches in the input. Their order has to match the order of the patch
            dimensions of `dictionary`.
        """
        super().__init__()
        self.dim = (dim,) if isinstance(dim, int) else dim
        patch_shape = tuple(dictionary.shape[1:])
        if len(self.dim) != len(patch_shape):
            raise ValueError(
                f'`dim` must have one entry per patch axis of the dictionary: got {len(self.dim)} entries '
                f'for patch shape {patch_shape}.'
            )
        if len(set(self.dim)) != len(self.dim):
            raise ValueError(f'`dim` must not contain duplicates, got {self.dim}.')

        # dim[k] is the input axis corresponding to dictionary axis k+1
        additional_info = {
            f'd{d}' if d >= 0 else f'dm{-d}': size for d, size in zip(self.dim, patch_shape, strict=True)
        }

        rearr_op = RearrangeOp(build_rearrange_and_flatten_pattern(self.dim), dict(additional_info))
        einsum_op = EinsumOp(
            dictionary.flatten(start_dim=1),
            einsum_rule='i j, ... j -> ... i',
        )
        self.dictionary_op = einsum_op @ rearr_op

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply dictionary analysis operation to input patches.

        The selected dimensions of the input patches are rearranged and flattened before applying the dictionary.

        Parameters
        ----------
        x
            Input tensor containing the patches.

        Returns
        -------
            Dictionary-transformed patches (coefficients).
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of AnalysisDictionaryOp.

        .. note::
            Prefer calling the instance of the AnalysisDictionaryOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self.dictionary_op(x)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply adjoint of AnalysisDictionaryOp.

        The adjoint operation applies the adjoint of the dictionary and reverses the rearrangement of the selected
        patch dimensions.

        Parameters
        ----------
        y
            Input tensor to be transformed using the adjoint operation.

        Returns
        -------
            Patches in the original dimension layout.
        """
        return self.dictionary_op.adjoint(y)
