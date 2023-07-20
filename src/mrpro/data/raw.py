"""MR raw data."""

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

import ismrmrd.xsd.ismrmrdschema.ismrmrd as ismrmrdschema
import torch


def _return_par_tensor(par, array_attr) -> torch.Tensor | None:
    if par is None:
        return None
    else:
        par_tensor = []
        for attr in array_attr:
            par_tensor.append(getattr(par, attr))
        return torch.tensor(par_tensor)


def return_par_matrix_tensor(
    par: ismrmrdschema.matrixSizeType,
) -> torch.Tensor | None:
    return _return_par_tensor(par, array_attr=('x', 'y', 'z'))


def return_par_enc_limits_tensor(
    par: ismrmrdschema.limitType,
) -> torch.Tensor | None:
    return _return_par_tensor(par, array_attr=('minimum', 'maximum', 'center'))


def return_acc_factor_tensor(
    par: ismrmrdschema.accelerationFactorType,
) -> torch.Tensor | None:
    return _return_par_tensor(par, array_attr=('kspace_encoding_step_1', 'kspace_encoding_step_2'))


# def bitmask_flag_to_strings(flag: int):
#     if flag > 0:
#         bmask = '{0:064b}'.format(flag)
#         bitmask_idx = [m.start() + 1 for m in re.finditer('1', bmask[::-1])]
#     else:
#         bitmask_idx = [
#             0,
#         ]
#     flag_strings = []
#     for knd in range(len(bitmask_idx)):
#         flag_strings.append(AcqFlags[bitmask_idx[knd]])
#     return flag_strings


def return_coil_label_dict(
    coil_label: list[ismrmrdschema.coilLabelType],
) -> dict:
    coil_label_dict = {}
    for idx, label in enumerate(coil_label):
        coil_label_dict[idx] = [label.coilNumber, label.coilName]
    return coil_label_dict
