"""Tests for the Constraints Operator."""

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

import pytest
import torch

from mrpro.operators import ConstraintsOp
from tests import RandomGenerator


@pytest.mark.parametrize(
    'bounds',
    [
        ((None, 1.0),),  # case (-infty, 0.)
        ((1.0, None),),  # case (1, \infty)
        ((-5.0, 5.0),),  # case (-5, 5)
    ],
)
def test_constraints_operator_bounds(bounds):

    random_generator = RandomGenerator(seed=0)

    # random tensor with arbitrary values
    # (make sure to hit bounds by scaling it with a "large" number)
    x = random_generator.float32_tensor(size=(36,), low=-100.0, high=100.0)

    # define constraints operator using the bounds
    Cop = ConstraintsOp(bounds)

    # transform tensor to be component-wise in the range defined by bounds
    (cx,) = Cop(x)
    ((a, b),) = bounds

    # check if min/max values of transformed tensor
    # correspond to the chosen bounds
    if a is not None and b is not None:  # case (a,b) with a<b
        torch.testing.assert_close(cx.min(), torch.tensor(a)) and torch.testing.assert_close(cx.max(), torch.tensor(b))
    elif a is not None and b is None:  # case (a, infty)
        torch.testing.assert_close(cx.min(), torch.tensor(a))
    elif a is None and b is not None:  # case (-infty, b)
        torch.testing.assert_close(cx.max(), torch.tensor(b))


@pytest.mark.parametrize(
    'bounds',
    [
        ((None, 1.0),),  # case (-infty, 0.)
        ((1.0, None),),  # case (1, \infty)
        ((-5.0, 5.0),),  # case (-5, 5)
        ((None, -1.0),),  # case (-infty, 0.)
        ((-1.0, None),),  # case (1, \infty)
    ],
)
def test_constraints_operator_inverse(bounds):
    """Tests if operator inverse inverser the operator."""

    random_generator = RandomGenerator(seed=0)

    # random tensor with arbitrary values
    x = random_generator.float32_tensor(size=(36,))

    # define constraints operator using the bounds
    Cop = ConstraintsOp(bounds)

    # transform tensor to be component-wise in the range defined by bounds
    (cx,) = Cop(x)

    # reverse transformation and check for equality
    (xx,) = Cop.inverse(cx)
    torch.testing.assert_close(xx, x)


@pytest.mark.parametrize(
    'bounds',
    [
        ((None, 1.0),),  # case (-infty, 0.)
        ((1.0, None),),  # case (1, \infty)
        ((-5.0, 5.0),),  # case (-5, 5)
        ((None, -1.0),),  # case (-infty, 0.)
        ((-1.0, None),),  # case (1, \infty)
    ],
)
def test_constraints_operator_no_nans(bounds):
    """Tests if the operator always returns valid values, never nans."""

    random_generator = RandomGenerator(seed=0)

    # random tensor with arbitrary values
    x = random_generator.float32_tensor(size=(36,), low=-100, high=100)

    # define constraints operator using the bounds
    Cop = ConstraintsOp(bounds)

    # transform tensor to be component-wise in the range defined by bounds
    (cx,) = Cop(x)

    # reverse transformation and check if no nans are returned
    (xx,) = Cop.inverse(cx)
    assert not torch.isnan(xx).any()


@pytest.mark.parametrize(
    'bounds',
    [
        (
            (None, None),
            (1.0, None),
            (None, 1.0),
        ),
    ],
)
def test_constraints_operator_multiple_inputs(bounds):

    random_generator = RandomGenerator(seed=0)

    # random tensors with arbitrary values
    x1 = random_generator.float32_tensor(size=(36, 72), low=-1, high=1)
    x2 = random_generator.float32_tensor(size=(36, 72), low=-1, high=1)
    x3 = random_generator.float32_tensor(size=(36, 72), low=-1, high=1)

    # define constraints operator using the bounds
    Cop = ConstraintsOp(bounds)

    # transform tensor to be component-wise in the range defined by bounds
    cx1, cx2, cx3 = Cop(x1, x2, x3)

    # reverse transformation and check if no nans are returned
    xx1, xx2, xx3 = Cop.inverse(cx1, cx2, cx3)
    torch.testing.assert_close(xx1, x1)
    torch.testing.assert_close(xx2, x2)
    torch.testing.assert_close(xx3, x3)


@pytest.mark.parametrize(
    'bounds',
    [
        ((1.0, -1.0), (-1.0, 1.0)),  # first one is invalid
        ((-1.0, 1.0), (1.0, -1.0)),  # second one is invalid
        ((1.0, 1.0),),
        ((torch.nan, 1),),
        ((-1, torch.nan),),
        ((torch.inf, -torch.inf),),
    ],
)
def test_constraints_operator_illegal_bounds(bounds):
    with pytest.raises(ValueError, match='invalid'):
        Cop = ConstraintsOp(bounds)
