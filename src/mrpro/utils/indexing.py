"""Indexer class for custom indexing with broadcasting."""

from collections.abc import Sequence
from typing import cast

import torch
import torch.testing

from mrpro.utils.reshape import reduce_view
from mrpro.utils.typing import TorchIndexerType


class Indexer:
    """Custom Indexing with broadcasting.

    This class is used to index tensors in a way that is consistent
    with the shape invariants of the data objects.

    On creation, an index and a shape are required.
    When calling the Indexer with a tensor, the tensor is first broadcasted to the shape,
    then the index is applied to the tensor.

    After indexing, remaining broadcasted dimensions are reduced to singleton dimensions again.
    Thus, using the same Indexer on tensors with different singleton dimensions will
    result in tensors with different shapes. All resulting tensors can be broadcasted
    to the shape that would result in indexing a full tensor already having the desired shape.

    Indexing never removes dimensions, and can only add new dimensions at the beginning of the tensor.

    The index can contain slices, integers, boolean masks, sequences of integers, and integer tensors:
    - Indexing with a slice
        Behaves like in numpy, always returns a view.
        Negative step sizes are not supported and will raise an IndexError.
        slice(None), i.e., means selecting the whole axis.
    - Indexing with an integer
        If the index is within the bounds of the broadcasted shape, indexing behaves like slicing with index:index+1.
        Otherwise, an IndexError is raised.
        Always returns a view.
    - Indexing with a boolean mask
        Singleton dimensions in the mask are interpreted as full slices. This matches broadcasting of the mask to
        the size of the respective axes of the tensor.
        If the mask has more than one non-singleton dimension, a new dimension is added at the beginning of the tensor,
        with length equal to the number of True values in the mask.
        At the indexed axes, singleton dimensions are kept.
        If the mask has only one non-singleton dimension, only the size of the indexed axes is changed.
        Only a single boolean mask is allowed, otherwise an IndexError is raised.
    - Indexing with a sequence of integers
        If a single indexer is a sequence of integers, the result is as if each value of the sequence was used as an
        integer index and the results were concatenated along the indexed dimension.
        If more than one sequence of integers is used, a new dimension at the beginning of the tensor,
        with the length equal to the shape of the sequences, is added. Indexed dimensions are kept as singleton.
        The different sequences must have the same shape, otherwise an IndexError is raised.
        Note that, as in numpy and torch, vectorized indexing is performed, not outer indexing.
    - None
        New axes can be added to the front of tensor by using None in the index.
        This is only allowed at the beginning of the index.
    - Ellipsis
        An indexing expression can contain a single ellipsis, which will be expanded to slice(None)
        for all axes that are not indexed.

    Implementation details:
    - On creation, the indexing expression is parsed and split into two parts: normal_index and fancy_index.
    - normal_index contains only indexing expressions that can be represented as view.
    - fancy_index contains all other indexing expressions.
    - On call
        - the tensor is broadcasted to the desired shape
        - the normal_index is applied.
        - if required, the fancy_index is applied.
        - remaining broadcasted dimensions are reduced to singleton dimensions.
    """

    def __init__(self, shape: tuple[int, ...], index: tuple[TorchIndexerType, ...]) -> None:
        """Initialize the Indexer.

        Parameters
        ----------
        shape
            broadcasted shape of the tensor to index. The tensor will be broadcasted to this shape.
        index
            The index to apply to the tensor.
        """
        normal_index: list[slice | int | None] = []
        """Used in phase 1 of the indexing, where we only consider integers and slices. Always does a view"""
        fancy_index: list[slice | torch.Tensor | tuple[int, ...] | None] = []
        """All non normal indices. Might not be possible to do a view."""
        has_fancy_index = False
        """Are there any advanced indices, such as boolean or integer array indices?"""
        vectorized_shape: None | tuple[int, ...] = None
        """Number of dimensions of the integer indices"""
        expanded_index: list[slice | torch.Tensor | tuple[int, ...] | None | int] = []
        """"Index with ellipsis expanded to full slices"""

        # basics checks and figuring out the number of axes already covered by the index,
        # which is needed to determine the number of axes that are covered by the ellipsis
        has_ellipsis = False
        has_boolean = False
        covered_axes = 0
        for idx_ in index:
            if idx_ is None:
                if has_ellipsis or covered_axes:
                    raise IndexError('New axes are only allowed at the beginning of the index')
            elif idx_ is Ellipsis:
                if has_ellipsis:
                    raise IndexError('Only one ellipsis is allowed')
                has_ellipsis = True
            elif isinstance(idx_, torch.Tensor) and idx_.dtype == torch.bool:
                if has_boolean:
                    raise IndexError('Only one boolean index is allowed')
                has_boolean = True
                covered_axes += idx_.ndim
            elif isinstance(idx_, int | slice | torch.Tensor) or (
                isinstance(idx_, Sequence) and all(isinstance(el, int) for el in idx_)
            ):
                covered_axes += 1
            else:
                raise IndexError(f'Unsupported index type {idx_}')

        if covered_axes > len(shape):
            raise IndexError('Too many indices. Indexing more than the number of available axes is not allowed')

        for idx_ in index:
            if idx_ is Ellipsis:
                # replacing ellipsis with full slices
                expanded_index.extend([slice(None)] * (len(shape) - covered_axes))
            elif isinstance(idx_, torch.Tensor | int | slice | None):
                expanded_index.append(idx_)
            else:  # must be Sequence[int], checked above
                # for consistency, we convert all non-tensor sequences of integers to tuples
                expanded_index.append(tuple(cast(Sequence[int], idx_)))

        if not has_ellipsis:
            # if there is no ellipsis, we interpret the index as if it was followed by ellipsis
            expanded_index.extend([slice(None)] * (len(shape) - covered_axes))

        number_of_vectorized_indices: int = 0
        shape_position: int = 0  # current position in the shape that we are indexing
        for idx in expanded_index:
            if idx is None:
                # we already checked that None is only allowed at the beginning of the index
                normal_index.append(None)
                fancy_index.append(slice(None))

            elif isinstance(idx, int):
                # always convert integers to slices
                if not -shape[shape_position] <= idx < shape[shape_position]:
                    raise IndexError(
                        f'Index {idx} out of bounds for axis {shape_position} with shape {shape[shape_position]}'
                    )
                normal_index.append(slice(idx, idx + 1))
                fancy_index.append(slice(None))
                shape_position += 1

            elif isinstance(idx, slice):
                if idx.step is not None and idx.step < 0:
                    raise IndexError('Negative step size for slices is not supported')
                normal_index.append(idx)
                fancy_index.append(slice(None))
                shape_position += 1
                continue

            elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
                # boolean indexing
                has_fancy_index = True

                while idx.ndim and idx.shape[0] == 1:
                    # remove leading singleton dimensions and replace by full slices
                    idx = idx.squeeze(0)
                    fancy_index.append(slice(None))
                    normal_index.append(slice(None))
                    shape_position += 1

                right_slice = []
                while idx.ndim and idx.shape[-1] == 1:
                    # remove trailing singleton dimensions and replace by full slices
                    idx = idx.squeeze(-1)
                    right_slice.append(slice(None))

                if idx.ndim == 1:
                    # single boolean dimension remains
                    fancy_index.extend(torch.nonzero(idx, as_tuple=True))
                    number_of_vectorized_indices += 1

                elif idx.ndim > 1:
                    # more than one non singleton dimension
                    for ids, idx_shape, data_shape in zip(
                        idx.nonzero(as_tuple=True),
                        idx.shape,
                        shape[shape_position : shape_position + idx.ndim],
                        strict=True,
                    ):
                        if idx_shape == 1:
                            # we interpret singleton dimensions as full slices
                            fancy_index.append(slice(None))
                        elif idx_shape != data_shape:
                            raise IndexError(
                                f'Boolean index has wrong shape, got {idx_shape} but expected {data_shape}'
                            )

                        else:
                            fancy_index.append(ids)
                            number_of_vectorized_indices += 1
                else:
                    # all singleton boolean mask
                    pass

                normal_index.extend(idx.ndim * [slice(None)])
                normal_index.extend(right_slice)
                fancy_index.extend(right_slice)
                shape_position += idx.ndim + len(right_slice)

            elif isinstance(idx, torch.Tensor) and idx.dtype in (
                torch.int64,  # long
                torch.int32,  # int
                torch.int16,
                torch.int8,
                torch.uint16,
                torch.uint32,
                torch.uint64,
            ):
                # integer array indexing
                if (idx >= shape[shape_position]).any() or (idx < -shape[shape_position]).any():
                    raise IndexError(
                        'Index out of bounds. '
                        f'Got values in the interval [{idx.min()}, {idx.max() + 1}) for axis {shape_position} '
                        f'with shape {shape[shape_position]}'
                    )
                if vectorized_shape is not None and vectorized_shape != idx.shape:
                    raise IndexError(
                        f'All vectorized indices must have the same shape. Got {idx.shape} and {vectorized_shape}'
                    )
                vectorized_shape = idx.shape
                has_fancy_index = True
                shape_position += 1
                number_of_vectorized_indices += 1
                normal_index.append(slice(None))
                fancy_index.append(idx.to(torch.int64))

            elif isinstance(idx, tuple):
                # integer Sequence
                if any(el >= shape[shape_position] or el < -shape[shape_position] for el in idx):
                    raise IndexError(
                        'Index out of bounds. '
                        f'Got values in the interval [{min(idx)}, {max(idx) + 1}) for axis {shape_position} '
                        f'with shape {shape[shape_position]}'
                    )
                if vectorized_shape is not None and vectorized_shape != (len(idx),):
                    raise IndexError('All vectorized indices must have the same shape')
                vectorized_shape = (len(idx),)
                has_fancy_index = True
                shape_position += 1
                normal_index.append(slice(None))
                fancy_index.append(idx)
                number_of_vectorized_indices += 1

            else:  # torch.Tensor
                raise IndexError(f'Unsupported index dtype {idx.dtype}')

        self.move_axes: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
        """final move-axes operation to move the vectorized indices to the beginning of the tensor"""
        self.more_than_one_vectorized_index = number_of_vectorized_indices > 1
        """there is more than one vectorized index, thus a new axis will be added"""

        if self.more_than_one_vectorized_index:
            # torch indexing would remove the dimensions, we want to keep them
            # as singleton dimension -> we need to add a new axis.
            # inserting it in between the indices forces the dimension added by
            # vectorized indices to be always at the beginning of the result.
            self.more_than_one_vectorized_index = True
            new_fancy_index = []
            for idx in fancy_index:
                new_fancy_index.append(idx)
                if isinstance(idx, torch.Tensor):
                    new_fancy_index.append(None)
                if isinstance(idx, tuple):
                    new_fancy_index.append(None)
            fancy_index = new_fancy_index

        elif vectorized_shape is not None and len(vectorized_shape) != 1:
            # for a single vectorized index, torch would insert it at the same position
            # this would shift the other axes, potentially causing violations of the shape invariants.
            # thus, we move the inserted axis to the beginning of the tensor, after axes inserted by None
            move_source_start = next(i for i, idx in enumerate(fancy_index) if isinstance(idx, torch.Tensor))
            move_source = tuple(range(move_source_start, move_source_start + len(vectorized_shape)))
            move_target_start = next(i for i, idx in enumerate(fancy_index) if idx is not None)
            move_target = tuple(range(move_target_start, move_target_start + len(vectorized_shape)))
            self.move_axes = (move_source, move_target)
            # keep a singleton axes at the indexed axis
            fancy_index.insert(move_source_start + 1, None)

        self.fancy_index = tuple(fancy_index) if has_fancy_index else ()
        self.normal_index = tuple(normal_index)
        self.shape = shape

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the index to a tensor."""
        try:
            tensor = tensor.broadcast_to(self.shape)
        except RuntimeError:
            raise IndexError('Tensor cannot be broadcasted to the desired shape') from None

        tensor = tensor[self.normal_index]  # will always be a view

        if not self.fancy_index:
            # nothing more to do
            tensor = reduce_view(tensor)
            return tensor

        # we need to modify the fancy index to efficiently handle broadcasted dimensions
        fancy_index: list[None | tuple[int, ...] | torch.Tensor | slice] = []
        tensor_index = 0
        stride = tensor.stride()
        for idx in self.fancy_index:
            if idx is None:
                fancy_index.append(idx)
                # don't increment tensor_index as this is a new axis
                continue
            if stride[tensor_index] == 0:
                # broadcasted dimension
                if isinstance(idx, slice):  # can only be slice(None) here
                    # collapse broadcasted dimensions to singleton, i.e. keep the dimension
                    fancy_index.append(slice(0, 1))
                elif not self.more_than_one_vectorized_index and isinstance(idx, tuple):
                    # as the dimension only exists due to broadcasting, it should be reduced to singleton
                    # there is already a None inserted after the index, so we don't need to keep the dimension
                    fancy_index.append((0,))
                elif not self.more_than_one_vectorized_index and isinstance(idx, torch.Tensor):
                    # same, but with more dimensions in the single vectorized index
                    # these axes will later be moved to the beginning of the tensor, as they would
                    # if the dimensions were not broadcasted
                    fancy_index.append(idx.new_zeros([1] * idx.ndim))
                else:
                    fancy_index.append(idx)
            else:
                fancy_index.append(idx)
            tensor_index += 1

        tensor = tensor[fancy_index]

        if self.move_axes[0]:
            # handle the special case of a single and integer index, where we need to move the new
            # axis to the beginning of the tensor
            tensor = tensor.moveaxis(self.move_axes[0], self.move_axes[1])

        return tensor
