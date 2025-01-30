"""Functions for tensor shaping, unit conversion, typing, etc."""

from mrpro.utils import slice_profiles
from mrpro.utils import typing
from mrpro.utils import unit_conversion
from mrpro.utils.fill_range import fill_range_
from mrpro.utils.smap import smap
from mrpro.utils.remove_repeat import remove_repeat
from mrpro.utils.zero_pad_or_crop import zero_pad_or_crop
from mrpro.utils.split_idx import split_idx
from mrpro.utils.reshape import broadcast_right, unsqueeze_left, unsqueeze_right, reduce_view, reshape_broadcasted
from mrpro.utils.TensorAttributeMixin import TensorAttributeMixin
__all__ = [
    "TensorAttributeMixin",
    "broadcast_right",
    "fill_range_",
    "reduce_view",
    "remove_repeat",
    "reshape_broadcasted",
    "slice_profiles",
    "smap",
    "split_idx",
    "typing",
    "unit_conversion",
    "unsqueeze_left",
    "unsqueeze_right",
    "zero_pad_or_crop"
]