"""Dictionary Operator."""

from collections.abc import Sequence

import torch

from mrpro.operators.EinsumOp import EinsumOp
from mrpro.operators.LinearOperator import LinearOperator
from mrpro.operators.RearrangeOp import RearrangeOp


class DictionaryOp(LinearOperator):
    """A Linear Operator that applies a dictionary to the given patches.

    Rearranges and flattens selected dimensions of the input patches and
    applies a dictionary using Einstein summation.
    """

    def __init__(
        self,
        dictionary: torch.Tensor,
        dim: tuple,
        patch_size: Sequence[int],
    ) -> None:
        """Initialize Dictionary Operator.

        Parameters
        ----------
        dictionary
            Dictionary tensor to be applied to the flattened patches.

        dim
            Dimensions of the patches that are moved to the last dimensions
            and flattened before applying the dictionary.

        patch_size
            Dimensions of the input patches tensor.
        """
        super().__init__()

        dim = tuple(sorted(i for i in dim if i >= 0) + sorted((i for i in dim if i < 0), reverse=True))

        all_dims = tuple(range(len(patch_size)))
        other_dims = tuple(dim_idx for dim_idx in all_dims if dim_idx not in [all_dims[d] for d in dim])
        dim_names = tuple(f'dim{dim_idx}' for dim_idx in all_dims)
        input_pattern = ' '.join(dim_names)
        other_pattern = ' '.join(dim_names[d] for d in other_dims)
        dimensions_pattern = ' '.join(dim_names[d] for d in dim)
        output_pattern = f'{other_pattern} ({dimensions_pattern})'
        additional_info = dict([(dim_names[d], patch_size[d]) for d in dim])

        rearr_op = RearrangeOp(
            f'{input_pattern} -> {output_pattern}',
            dict(additional_info),
        )

        einsum_op = EinsumOp(
            dictionary,
            einsum_rule='i j, ... j -> ... i',
        )

        self.dictionary_op = einsum_op @ rearr_op

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply dictionary operation to input patches.

        The selected dimensions of the input patches are rearranged and
        flattened before applying the dictionary.

        Parameters
        ----------
        x
            Input tensor containing the patches.

        Returns
        -------
        tuple[torch.Tensor]
            Dictionary-transformed patches.
        """
        return super().__call__(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply forward of DictionaryOp."""
        y = self.dictionary_op(x)
        return y

    def adjoint(self, y: torch.Tensor) -> tuple[torch.Tensor]:
        """Apply adjoint of DictionaryOp.

        The adjoint operation applies the adjoint of the dictionary and
        reverses the rearrangement of the selected patch dimensions.

        Parameters
        ----------
        y
            Input tensor to be transformed using the adjoint operation.

        Returns
        -------
        tuple[torch.Tensor]
            Patches in the original dimension layout.
        """
        x = self.dictionary_op.adjoint(y)
        return x
