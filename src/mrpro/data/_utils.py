"""Utility Functions for Data Handling."""

# Copyright 2023 Physikalisch-Technische Bundesanstalt
#
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import annotations

import functools

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema


def return_coil_label_dict(
    coil_label: list[ismrmrdschema.coilLabelType],
) -> dict:
    coil_label_dict = {}
    for idx, label in enumerate(coil_label):
        coil_label_dict[idx] = [label.coilNumber, label.coilName]
    return coil_label_dict


def rgetattr(obj, attr, *args):
    """
    Recursive getattr for nested attributes.
    Parameters
    
    """
    return functools.reduce(
        lambda obj, attr: getattr(obj, attr, *args),
        [obj] + attr.split('.'),
    )
