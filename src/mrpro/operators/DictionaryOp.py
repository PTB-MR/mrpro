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


class DictionaryOp(LinearOperator):
    """A Linear Operator that applies a dictionary to the given patches.

    Rearranges and flattens selected dimensions of the input patches and applies a dictionary using Einstein summation.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        dim: Sequence[int] | int,
    ) -> None:
        """Initialize Dictionary Operator.

        Parameters
        ----------
        dictionary
            Dictionary tensor to be applied to the flattened patches.
            Shape e.g. `(n_entries, d1, d0)` for 2D patches or for 3D patches `(n_entries, d3, d1, d0)`

        dim
            Dimensions of the patches.

        """
        super().__init__()
        self.dim = (dim,) if isinstance(dim, int) else dim
        patch_size = dictionary.shape[1:]
        additional_info = {f'd{d}' if d >= 0 else f'dm{-d}': patch_size[d] for d in self.dim}

        rearr_op = RearrangeOp(
            build_rearrange_and_flatten_pattern(self.dim),
            dict(additional_info),
        )

        einsum_op = EinsumOp(
            dictionary.flatten(start_dim=1),
            einsum_rule='i j, ... j -> ... i',
        )

        self.dictionary_op = einsum_op @ rearr_op

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply dictionary operation to input patches.

        The selected dimensions of the input patches are rearranged and flattened before applying the dictionary.

        Parameters
        ----------
        x
            Input tensor containing the patches.

        Returns
        -------
            Dictionary-transformed patches.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of DictionaryOp.

        .. note::
            Prefer calling the instance of the DictionaryOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        y = self.dictionary_op(x)
        return y

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply adjoint of DictionaryOp.

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
        x = self.dictionary_op.adjoint(y)
        return x
