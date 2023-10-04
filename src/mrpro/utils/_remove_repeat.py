"""remove_repeat utility function"""

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

import torch


def remove_repeat(tensor: torch.Tensor, tol: float) -> torch.Tensor:
    """Replace dimensions with all equal values with singletons.

    Parameters
    ----------
    tensor:
        The tensor. Must be real
    tol:
        The tolerance
    """

    def can_be_singleton(dim: int) -> bool:
        # If the distance between min and max is smaller than the tolerance, all values are the same.
        return bool(torch.all((tensor.amax(dim=dim) - tensor.amin(dim=dim)) <= tol).item())

    take_first = slice(0, 1)
    take_all = slice(None)
    index = tuple(take_first if can_be_singleton(dim) else take_all for dim in range(tensor.ndim))
    return tensor[index]
