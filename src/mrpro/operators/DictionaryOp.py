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


class DictionaryAnalysisOp(LinearOperator):
    """Dictionary analysis operator (patches -> coefficients).

    Rearranges and flattens the selected patch dimensions of the input and computes the inner product with
    every dictionary entry, i.e. ``c_i = <d_i, x> = sum_j conj(d_i[j]) x[j]``.
    The adjoint is the synthesis ``x = sum_i c_i d_i``.
    """

    def __init__(self, dictionary: torch.Tensor, dim: Sequence[int] | int) -> None:
        """Initialize Dictionary Operator.

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
            dictionary.flatten(start_dim=1).conj(),
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
        """Apply forward of DictionaryAnalysisOp.

        .. note::
            Prefer calling the instance of the DictionaryAnalysisOp operator as ``operator(x)`` over
            directly calling this method. See this PyTorch `discussion <https://discuss.pytorch.org/t/is-model-forward-x-the-same-as-model-call-x/33460/3>`_.
        """
        return self.dictionary_op(x)

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply adjoint of DictionaryAnalysisOp.

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


class DictionarySynthesisOp(LinearOperator):
    """Dictionary synthesis operator (coefficients -> patches).

    Computes ``x = sum_i c_i d_i`` and places the result in the original patch layout.
    It is the adjoint of `DictionaryAnalysisOp`, and vice versa.
    """

    def __init__(self, dictionary: torch.Tensor, dim: Sequence[int] | int) -> None:
        """Initialize Dictionary Synthesis Operator.

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
        self._analysis = DictionaryAnalysisOp(dictionary, dim)

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
        """Apply forward of DictionarySynthesisOp.

        .. note::
            Prefer calling the instance of the DictionarSynthesisOp operator as ``operator(x)`` over
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
