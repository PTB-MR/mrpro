"""Tests for join modules."""

from typing import Literal

import pytest
import torch
from mr2.nn.join import Add, Concat
from mr2.utils import RandomGenerator


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('mode', 'input_shapes', 'expected_shape'),
    [
        ('crop', [(1, 3, 32, 32), (1, 5, 30, 30)], (1, 8, 30, 30)),
        ('zero', [(1, 3, 32, 32), (1, 5, 34, 34)], (1, 8, 34, 34)),
        ('linear', [(1, 3, 32, 32), (1, 5, 34, 34)], (1, 8, 34, 34)),
        ('nearest', [(1, 3, 32, 32), (1, 5, 34, 34)], (1, 8, 34, 34)),
    ],
)
def test_concat_basic(
    mode: Literal['crop', 'zero', 'replicate', 'circular', 'linear', 'nearest'],
    input_shapes: list[tuple[int, ...]],
    expected_shape: tuple[int, ...],
    device: str,
) -> None:
    """Test Concat basic functionality."""
    rng = RandomGenerator(seed=42)
    xs = [rng.float32_tensor(shape).to(device).requires_grad_(True) for shape in input_shapes]
    concat = Concat(mode=mode).to(device)

    output = concat(*xs)
    assert output.shape == expected_shape
    assert not output.isnan().any(), 'NaN values in output'

    output.sum().backward()
    for x in xs:
        assert x.grad is not None, 'No gradient computed for input'
        assert not x.grad.isnan().any(), 'NaN values in input gradients'


@pytest.mark.parametrize(
    'device',
    [
        pytest.param('cpu', id='cpu'),
        pytest.param('cuda', id='cuda', marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    ('mode', 'input_shapes', 'expected_shape'),
    [
        ('crop', [(1, 3, 32, 32), (1, 3, 30, 30)], (1, 3, 30, 30)),
        ('zero', [(1, 3, 32, 32), (1, 3, 34, 34)], (1, 3, 34, 34)),
        ('replicate', [(1, 1, 1, 2), (1, 1, 1, 3)], (1, 1, 1, 3)),
        ('circular', [(1, 1, 1, 2), (1, 1, 1, 4)], (1, 1, 1, 4)),
    ],
)
def test_add_basic(
    mode: Literal['crop', 'zero', 'replicate', 'circular'],
    input_shapes: list[tuple[int, ...]],
    expected_shape: tuple[int, ...],
    device: str,
) -> None:
    """Test Add basic functionality."""
    rng = RandomGenerator(seed=42)
    xs = [rng.float32_tensor(shape).to(device).requires_grad_(True) for shape in input_shapes]
    add = Add(mode=mode).to(device)

    output = add(*xs)
    assert output.shape == expected_shape
    assert not output.isnan().any(), 'NaN values in output'

    output.sum().backward()
    for x in xs:
        assert x.grad is not None, 'No gradient computed for input'
        assert not x.grad.isnan().any(), 'NaN values in input gradients'


@pytest.mark.parametrize(
    ('dim', 'input_shapes', 'expected_shape'),
    [
        (0, [(1, 3, 32, 32), (1, 3, 32, 32)], (2, 3, 32, 32)),
        (1, [(1, 3, 32, 32), (1, 5, 32, 32)], (1, 8, 32, 32)),
        (2, [(1, 3, 32, 32), (1, 3, 32, 32)], (1, 3, 64, 32)),
    ],
)
def test_concat_dimensions(dim: int, input_shapes: list[tuple[int, ...]], expected_shape: tuple[int, ...]) -> None:
    """Test Concat with different concatenation dimensions."""
    rng = RandomGenerator(seed=42)
    xs = [rng.float32_tensor(shape).requires_grad_(True) for shape in input_shapes]
    concat = Concat(mode='fail', dim=dim)
    output = concat(*xs)
    assert output.shape == expected_shape


def test_concat_values() -> None:
    """Test that Concat preserves input values correctly."""
    x1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).requires_grad_(True)
    x2 = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]).requires_grad_(True)

    concat = Concat(mode='fail')
    output = concat(x1, x2)

    expected = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]])
    torch.testing.assert_close(output, expected)


def test_add_values() -> None:
    """Test that Add correctly sums input values."""
    x1 = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]]).requires_grad_(True)
    x2 = torch.tensor([[[[5.0, 6.0], [7.0, 8.0]]]]).requires_grad_(True)

    add = Add(mode='fail')
    output = add(x1, x2)

    expected = torch.tensor([[[[6.0, 8.0], [10.0, 12.0]]]])
    torch.testing.assert_close(output, expected)


def test_concat_mode_fail() -> None:
    """Test Concat with mode='fail'."""
    rng = RandomGenerator(seed=42)

    x1 = rng.float32_tensor((1, 3, 32, 32))
    x2 = rng.float32_tensor((1, 5, 32, 32))
    concat = Concat(mode='fail')
    output = concat(x1, x2)
    assert output.shape == (1, 8, 32, 32)

    x3 = rng.float32_tensor((1, 3, 30, 30))
    with pytest.raises(RuntimeError):
        concat(x1, x3)


def test_add_mode_fail() -> None:
    """Test Add with mode='fail'."""
    rng = RandomGenerator(seed=42)

    x1 = rng.float32_tensor((1, 3, 32, 32))
    x2 = rng.float32_tensor((1, 3, 32, 32))
    add = Add(mode='fail')
    output = add(x1, x2)
    assert output.shape == (1, 3, 32, 32)

    x3 = rng.float32_tensor((1, 3, 30, 30))
    with pytest.raises(RuntimeError):
        add(x1, x3)


@pytest.mark.parametrize('module_class', [Concat, Add])
def test_invalid_mode(module_class: type) -> None:
    """Test modules with invalid mode."""
    with pytest.raises(ValueError, match='mode must be one of'):
        module_class(mode='invalid_mode')
