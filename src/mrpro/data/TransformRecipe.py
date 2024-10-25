import itertools
from collections import OrderedDict
from dataclasses import dataclass

import einops
import einops.parsing
from einops.einops import _expected_axis_length, _unknown_axis_length


@dataclass
class TransformRecipe:
    """Recipe describing an einops transformation."""

    elementary_axes_lengths: list[int]
    """list of sizes for elementary axes as they appear in the left expression.
    This is the shape after the first transposition, excluding any ellipsis dimensions."""

    axis_name2elementary_axis: dict[str, int]
    """Mapping from axis name to position."""

    input_composition_known_unknown: list[tuple[list[int], list[int]]]
    """Each dimension in input can help to reconstruct the length of one elementary axis
    or verify one of the dimensions. Each element points to an element of elementary_axes_lengths."""

    axes_permutation: list[int]
    """Permutation applied to elementary axes, if ellipsis is absent."""

    first_reduced_axis: int
    """Position of the first reduced axis after permutation."""

    added_axes: dict[int, int]
    """Mapping of positions to elementary axes that should appear at those positions."""

    output_composite_axes: list[list[int]]
    """IDs of axes as they appear in the result, pointing to elementary_axes_lengths,
    used to infer result dimensions."""

    @classmethod
    def prepare_transformation_recipe(
        cls,
        pattern: str,
        axes_names: tuple[str, ...],
        ndim: int,
    ) -> TransformRecipe:
        """Perform initial parsing of pattern and provided supplementary info."""
        _ellipsis: str = '…'  # this is a single unicode symbol.
        left_str, right_str = pattern.split('->')
        left = einops.parsing.ParsedExpression(left_str)
        right = einops.parsing.ParsedExpression(right_str)

        if not left.has_ellipsis and right.has_ellipsis:
            raise RearrangeError(f'Ellipsis found in right side, but not left side of a pattern {pattern}')
        if left.has_ellipsis and left.has_ellipsis_parenthesized:
            raise RearrangeError(f'Ellipsis inside parenthesis in the left side is not allowed: {pattern}')

        difference = set.difference(left.identifiers, right.identifiers)
        if len(difference) > 0:
            raise RearrangeError(f'Unexpected identifiers on the left side of pattern: {difference}')
        axes_without_size = set.difference(
            {ax for ax in right.identifiers if not isinstance(ax, einops.parsing.AnonymousAxis)},
            {*left.identifiers, *axes_names},
        )

        if len(axes_without_size) > 0:
            raise RearrangeError(f'Specify sizes for new axes in pattern: {axes_without_size}')

        if left.has_ellipsis:
            n_other_dims = len(left.composition) - 1
            if ndim < n_other_dims:
                raise RearrangeError(f'Wrong shape: expected >={n_other_dims} dims. Received {ndim}-dim tensor.')
            ellipsis_ndim = ndim - n_other_dims
            ell_axes = [_ellipsis + str(i) for i in range(ellipsis_ndim)]
            left_composition = []
            for composite_axis in left.composition:
                if composite_axis == _ellipsis:
                    for axis in ell_axes:
                        left_composition.append([axis])
                else:
                    left_composition.append(composite_axis)

            right_composition = []
            for composite_axis in right.composition:
                if composite_axis == _ellipsis:
                    for axis in ell_axes:
                        right_composition.append([axis])
                else:
                    group = []
                    for axis in composite_axis:
                        if axis == _ellipsis:
                            group.extend(ell_axes)
                        else:
                            group.append(axis)
                    right_composition.append(group)

            left.identifiers.update(ell_axes)
            left.identifiers.remove(_ellipsis)
            if right.has_ellipsis:
                right.identifiers.update(ell_axes)
                right.identifiers.remove(_ellipsis)
        else:
            if ndim != len(left.composition):
                raise RearrangeError(f'Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.')
            left_composition = left.composition
            right_composition = right.composition

        # parsing all dimensions to find out lengths
        axis_name2known_length: dict[str | einops.parsing.AnonymousAxis, int] = OrderedDict()
        for composite_axis in left_composition:
            for axis_name in composite_axis:
                if isinstance(axis_name, einops.parsing.AnonymousAxis):
                    axis_name2known_length[axis_name] = axis_name.value
                else:
                    axis_name2known_length[axis_name] = _unknown_axis_length

        # axis_ids_after_first_reshape = range(len(axis_name2known_length)) at this point

        repeat_axes_names = []
        for axis_name in right.identifiers:
            if axis_name not in axis_name2known_length:
                if isinstance(axis_name, einops.parsing.AnonymousAxis):
                    axis_name2known_length[axis_name] = axis_name.value
                else:
                    axis_name2known_length[axis_name] = _unknown_axis_length
                repeat_axes_names.append(axis_name)

        axis_name2position = {name: position for position, name in enumerate(axis_name2known_length)}

        # axes provided as kwargs
        for elementary_axis in axes_names:
            if not einops.parsing.ParsedExpression.check_axis_name(elementary_axis):
                raise RearrangeError('Invalid name for an axis', elementary_axis)
            if elementary_axis not in axis_name2known_length:
                raise RearrangeError(f'Axis {elementary_axis} is not used in transform')
            axis_name2known_length[elementary_axis] = _expected_axis_length

        input_axes_known_unknown = []
        # some shapes are inferred later - all information is prepared for faster inference
        for i, composite_axis in enumerate(left_composition):
            known: set[str] = {axis for axis in composite_axis if axis_name2known_length[axis] != _unknown_axis_length}
            unknown: set[str] = {
                axis for axis in composite_axis if axis_name2known_length[axis] == _unknown_axis_length
            }
            if len(unknown) > 1:
                raise RearrangeError(f'Could not infer sizes for {unknown}')
            assert len(unknown) + len(known) == len(composite_axis)
            input_axes_known_unknown.append(
                ([axis_name2position[axis] for axis in known], [axis_name2position[axis] for axis in unknown])
            )

        axis_position_after_reduction: dict[str, int] = {}
        for axis_name in itertools.chain(*left_composition):
            if axis_name in right.identifiers:
                axis_position_after_reduction[axis_name] = len(axis_position_after_reduction)

        result_axes_grouping: list[list[int]] = [
            [axis_name2position[axis] for axis in composite_axis] for i, composite_axis in enumerate(right_composition)
        ]

        ordered_axis_left = list(itertools.chain(*left_composition))
        ordered_axis_right = list(itertools.chain(*right_composition))
        reduced_axes = [axis for axis in ordered_axis_left if axis not in right.identifiers]
        order_after_transposition = [axis for axis in ordered_axis_right if axis in left.identifiers] + reduced_axes
        axes_permutation = [ordered_axis_left.index(axis) for axis in order_after_transposition]
        added_axes = {
            i: axis_name2position[axis_name]
            for i, axis_name in enumerate(ordered_axis_right)
            if axis_name not in left.identifiers
        }

        first_reduced_axis = len(order_after_transposition) - len(reduced_axes)

        return cls(
            elementary_axes_lengths=list(axis_name2known_length.values()),
            axis_name2elementary_axis={axis: axis_name2position[axis] for axis in axes_names},
            input_composition_known_unknown=input_axes_known_unknown,
            axes_permutation=axes_permutation,
            first_reduced_axis=first_reduced_axis,
            added_axes=added_axes,
            output_composite_axes=result_axes_grouping,
        )


class RearrangeError(ValueError):
    """Error in rearrange operation."""
