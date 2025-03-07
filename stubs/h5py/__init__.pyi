from . import h5a as h5a, h5d as h5d, h5ds as h5ds, h5f as h5f, h5fd as h5fd, h5g as h5g, h5p as h5p, h5pl as h5pl, h5r as h5r, h5s as h5s, h5t as h5t, h5z as h5z
from ._hl import filters as filters
from ._hl.attrs import AttributeManager as AttributeManager
from ._hl.base import Empty as Empty, HLObject as HLObject, is_hdf5 as is_hdf5
from ._hl.dataset import Dataset as Dataset
from ._hl.datatype import Datatype as Datatype
from ._hl.files import File as File, register_driver as register_driver, registered_drivers as registered_drivers, unregister_driver as unregister_driver
from ._hl.group import ExternalLink as ExternalLink, Group as Group, HardLink as HardLink, SoftLink as SoftLink
from ._hl.vds import VirtualLayout as VirtualLayout, VirtualSource as VirtualSource
from ._selector import MultiBlockSlice as MultiBlockSlice
from .h5 import get_config as get_config
from .h5r import Reference as Reference, RegionReference as RegionReference
from .h5s import UNLIMITED as UNLIMITED
from .h5t import check_dtype as check_dtype, check_enum_dtype as check_enum_dtype, check_opaque_dtype as check_opaque_dtype, check_ref_dtype as check_ref_dtype, check_string_dtype as check_string_dtype, check_vlen_dtype as check_vlen_dtype, enum_dtype as enum_dtype, opaque_dtype as opaque_dtype, ref_dtype as ref_dtype, regionref_dtype as regionref_dtype, special_dtype as special_dtype, string_dtype as string_dtype, vlen_dtype as vlen_dtype

def run_tests(args: str = ''): ...
def enable_ipython_completer(): ...
