"""Modify AcqInfo."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from collections.abc import Callable

import torch

from mrpro.data.AcqInfo import AcqInfo


def modify_acq_info(fun_modify: Callable, acq_info: AcqInfo) -> AcqInfo:
    """Go through all fields of AcqInfo object and apply changes.

    Parameters
    ----------
    fun_modify
        Function which takes AcqInfo fields as input and returns modified AcqInfo field
    acq_info
        AcqInfo object
    """
    # Apply function to all fields of acq_info
    for field in dataclasses.fields(acq_info):
        current = getattr(acq_info, field.name)
        if isinstance(current, torch.Tensor):
            setattr(acq_info, field.name, fun_modify(current))
        elif dataclasses.is_dataclass(current):
            for subfield in dataclasses.fields(current):
                subcurrent = getattr(current, subfield.name)
                setattr(current, subfield.name, fun_modify(subcurrent))

    return acq_info
