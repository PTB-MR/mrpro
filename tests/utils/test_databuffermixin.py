"""Tests for the operators module."""

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

from mrpro.data import Data
from mrpro.utils import DataBufferMixin


class DataObject(Data):
    """A Data Class"""

    def __init__():
        self.data = torch.tensor(1.0)


class Module(DataBufferMixin):
    """A module using the mixin"""

    def __init__(self):
        super().__init__()
        self.register_buffer('data', DataObject())


def test_register_buffer():
    obj = Module()
    assert 'data' in obj._data

    # A tensor should still be registered as a buffer
    obj.register_buffer('data2', torch.tensor(1.0))
    assert 'data2' not in obj._data
    assert 'data2' in obj._buffer

    assert hasattr(obj, 'data')
    assert hasattr(obj, 'data2')


def test_state_dict():
    obj = Module()
    assert 'data' in obj.state_dict()


def test_apply():
    def set_to_zero(obj):
        if isinstance(obj, Data):
            obj.data[:] = 0

    obj = Module()
    obj._apply(set_to_zero)
    assert obj.data == 0.0


def test_delete():
    obj = Module()
    del obj.data
    assert not hasattr(obj, 'data')


def test_setgetattr():
    obj = Module()
    other_data = DataObject()
    other_data.data[:] = 2.0
    obj.data = other_data
    assert obj.data is other_data


def test_error_messages():
    obj = Module()
    data = DataObject()
    with pytest.raises(TypeError):
        obj.register_buffer(1, data)
    with pytest.raises(KeyError):
        obj.register_buffer('a.a', data)
    with pytest.raises(KeyError):
        obj.register_buffer('', data)
    with pytest.raises(KeyError):
        obj.register_buffer('data', data)


def test_to():
    obj = Module()
    obj.to(dtype=torch.float64)
    assert obj.data.data.dtype is torch.float64
