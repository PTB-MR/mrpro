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
    """A Data Class."""

    data: torch.Tensor
    header: None


class Module(DataBufferMixin, torch.nn.Module):
    """A module using the mixin."""

    def __init__(self):
        super().__init__()
        self.register_buffer('data', DataObject(torch.tensor(1.0), None))


def test_databuffer_register_buffer():
    obj = Module()
    assert 'data' in obj._data
    # A tensor should still be registered as a buffer
    obj.register_buffer('data2', torch.tensor(1.0), None)
    assert 'data2' not in obj._data
    assert 'data2' in obj._buffers

    assert hasattr(obj, 'data')
    assert hasattr(obj, 'data2')
    assert isinstance(obj.data, Data)


@pytest.mark.skip(reason='not implemented')
def test_databuffer_state_dict():
    # TODO: Needs an implementation
    obj = Module()
    assert 'data' in obj.state_dict()


def test_databuffer_apply():
    def set_to_zero(obj):
        print(obj)
        if isinstance(obj, Data):
            obj.data *= 0

    obj = Module()
    print(obj.data)
    obj._apply(set_to_zero)
    print(obj.data)
    assert obj.data.data == 0.0


def test_databuffer_delete():
    obj = Module()
    del obj.data
    assert not hasattr(obj, 'data')


def test_databuffer_setgetattr():
    obj = Module()
    other_data = DataObject(torch.tensor(2.0), None)
    obj.data = other_data
    assert obj.data is other_data


def test_databuffer_error_messages():
    obj = Module()
    data = DataObject(torch.tensor(1.0), None)
    with pytest.raises(TypeError):
        obj.register_buffer(1, data)
    with pytest.raises(KeyError):
        obj.register_buffer('a.a', data)
    with pytest.raises(KeyError):
        obj.register_buffer('', data)
    with pytest.raises(KeyError):
        obj.register_buffer('data', data)


def test_databuffer_to():
    obj = Module()
    obj.to(dtype=torch.float64)
    assert obj.data.data.dtype is torch.float64
