"""Optimizer Status base class."""

# Copyright 2024 Physikalisch-Technische Bundesanstalt
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

from typing import TypedDict

import torch


class OptimizerStatus(TypedDict):
    """Base class for OptimizerStatus."""

    solution: tuple[torch.Tensor, ...]
    """Current estimate(s) of the solution. """

    iteration_number: int
    """Current iteration of the (iterative) algorithm."""
