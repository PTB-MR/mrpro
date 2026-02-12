from collections.abc import Sequence

import torch
from mr2.utils import RandomGenerator


def test_bool() -> None:
    """Test bool method for valid boolean return."""
    rng = RandomGenerator(seed=42)
    value: bool = rng.bool()
    assert isinstance(value, bool)


def test_float32(low: float = 0.0, high: float = 1.0) -> None:
    """Test float32 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: float = rng.float32(low=low, high=high)
    assert low <= value < high


def test_float64(low: float = 0.0, high: float = 1.0) -> None:
    """Test float64 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: float = rng.float64(low=low, high=high)
    assert low <= value < high


def test_complex64(low: float = 0.0, high: float = 1.0) -> None:
    """Test complex64 method for amplitude bounds."""
    rng = RandomGenerator(seed=42)
    value: complex = rng.complex64(low=low, high=high)
    amplitude = (value.real**2 + value.imag**2) ** 0.5
    assert low <= amplitude < high


def test_complex128(low: float = 0.0, high: float = 1.0) -> None:
    """Test complex128 method for amplitude bounds."""
    rng = RandomGenerator(seed=42)
    value: complex = rng.complex128(low=low, high=high)
    amplitude = (value.real**2 + value.imag**2) ** 0.5
    assert low <= amplitude < high


def test_int8(low: int = -128, high: int = 127) -> None:
    """Test int8 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.int8(low=low, high=high)
    assert low <= value < high


def test_int16(low: int = -1 << 15, high: int = 1 << 15) -> None:
    """Test int16 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.int16(low=low, high=high)
    assert low <= value < high


def test_int32(low: int = -1 << 31, high: int = 1 << 31) -> None:
    """Test int32 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.int32(low=low, high=high)
    assert low <= value < high


def test_int64(low: int = -1 << 63, high: int = (1 << 63) - 1) -> None:
    """Test int64 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.int64(low=low, high=high)
    assert low <= value < high


def test_uint8(low: int = 0, high: int = 1 << 8) -> None:
    """Test uint8 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.uint8(low=low, high=high)
    assert low <= value < high


def test_uint16(low: int = 0, high: int = 1 << 16) -> None:
    """Test uint16 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.uint16(low=low, high=high)
    assert low <= value < high


def test_uint32(low: int = 0, high: int = 1 << 32) -> None:
    """Test uint32 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.uint32(low=low, high=high)
    assert low <= value < high


def test_uint64(low: int = 0, high: int = (1 << 64) - 1) -> None:
    """Test uint64 method for bounds."""
    rng = RandomGenerator(seed=42)
    value: int = rng.uint64(low=low, high=high)
    assert low <= value < high


def test_float32_tensor(size: Sequence[int] = (3, 2), low: float = 0.0, high: float = 1.0) -> None:
    """Test float32_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.float32_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor < high)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_float64_tensor(size: Sequence[int] = (3,), low: float = 0.0, high: float = 1.0) -> None:
    """Test float64_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.float64_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor < high)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_complex64_tensor(size: Sequence[int] = (2, 2), low: float = 0.0, high: float = 1.0) -> None:
    """Test complex64_tensor for shape, amplitude bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.complex64_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    amplitudes = torch.sqrt(tensor.real**2 + tensor.imag**2)
    assert torch.all(low <= amplitudes)
    assert torch.all(amplitudes < high)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor.imag)) > 1


def test_complex128_tensor(size: Sequence[int] = (1, 2, 3), low: float = 0.0, high: float = 1.0) -> None:
    """Test complex128_tensor for shape, amplitude bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.complex128_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    amplitudes = torch.sqrt(tensor.real**2 + tensor.imag**2)
    assert torch.all(low <= amplitudes)
    assert torch.all(amplitudes < high)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor.imag)) > 1


def test_int8_tensor(size: Sequence[int] = (4,), low: int = -128, high: int = 128) -> None:
    """Test int8_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.int8_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor <= high - 1)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_int16_tensor(size: Sequence[int] = (4,), low: int = -1 << 15, high: int = 1 << 15) -> None:
    """Test int16_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.int16_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor <= high - 1)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_int32_tensor(size: Sequence[int] = (4,), low: int = -1 << 31, high: int = 1 << 31) -> None:
    """Test int32_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.int32_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor <= high - 1)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_int64_tensor(
    size: Sequence[int] = (4,),
    low: int = -1 << 63,
    high: int = 1 << 63 - 1,  # -1 due to https://github.com/pytorch/pytorch/issues/81446
) -> None:
    """Test int64_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.int64_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor <= high - 1)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_uint8_tensor(size: Sequence[int] = (4,), low: int = 0, high: int = 256) -> None:
    """Test uint8_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.uint8_tensor(size=size, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor <= high - 1)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_bool_tensor(size: Sequence[int] = (16,)) -> None:
    """Test bool_tensor for shape and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.bool_tensor(size=size)
    assert tensor.shape == size
    assert tensor.any()
    assert not tensor.all()


def test_float32_tuple(size: int = 5, low: float = 0.0, high: float = 1.0) -> None:
    """Test float32_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.float32_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_float64_tuple(size: int = 5, low: float = 0.0, high: float = 1.0) -> None:
    """Test float64_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.float64_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_complex64_tuple(size: int = 5, low: float = 0.0, high: float = 1.0) -> None:
    """Test complex64_tuple for length, amplitude bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.complex64_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    amplitudes = [(x.real**2 + x.imag**2) ** 0.5 for x in tuple_values]
    assert all(low <= amp < high for amp in amplitudes)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_complex128_tuple(size: int = 5, low: float = 0.0, high: float = 1.0) -> None:
    """Test complex128_tuple for length, amplitude bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.complex128_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    amplitudes = [(x.real**2 + x.imag**2) ** 0.5 for x in tuple_values]
    assert all(low <= amp < high for amp in amplitudes)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_uint8_tuple(size: int = 5, low: int = 0, high: int = 255) -> None:
    """Test uint8_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.uint8_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_uint16_tuple(size: int = 5, low: int = 0, high: int = 1 << 16) -> None:
    """Test uint16_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.uint16_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_uint32_tuple(size: int = 5, low: int = 0, high: int = 1 << 32) -> None:
    """Test uint32_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.uint32_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_uint64_tuple(size: int = 5, low: int = 0, high: int = (1 << 64) - 1) -> None:
    """Test uint64_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.uint64_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_int8_tuple(size: int = 5, low: int = -128, high: int = 128) -> None:
    """Test int8_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.int8_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x <= high - 1 for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_int16_tuple(size: int = 5, low: int = -1 << 15, high: int = 1 << 15) -> None:
    """Test int16_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.int16_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x <= high - 1 for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_int32_tuple(size: int = 5, low: int = -1 << 31, high: int = 1 << 31) -> None:
    """Test int32_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.int32_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x <= high - 1 for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_int64_tuple(size: int = 5, low: int = -1 << 63, high: int = (1 << 63) - 1) -> None:
    """Test int64_tuple for length, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tuple_values = rng.int64_tuple(size=size, low=low, high=high)
    assert len(tuple_values) == size
    assert all(low <= x < high for x in tuple_values)
    if size > 1:
        assert len(set(tuple_values)) > 1


def test_ascii(size: int = 10) -> None:
    """Test ascii for length and character range."""
    rng = RandomGenerator(seed=42)
    string = rng.ascii(size=size)
    assert len(string) == size
    assert all(32 <= ord(c) < 127 for c in string)


def test_rand_like(
    shape: Sequence[int] = (2, 3), dtype: torch.dtype = torch.float64, low: float = 0.0, high: float = 1.0
) -> None:
    """Test rand_like for shape, bounds, and value differences."""
    x = torch.zeros(shape, dtype=dtype)
    rng = RandomGenerator(seed=42)
    tensor = rng.rand_like(x, low=low, high=high)
    assert tensor.shape == x.shape
    assert torch.all(low <= tensor)
    assert torch.all(tensor < high)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_rand_tensor(
    size: Sequence[int] = (2, 3), dtype: torch.dtype = torch.float32, low: float = 0, high: float = 1
) -> None:
    """Test rand_tensor for shape, bounds, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.rand_tensor(size=size, dtype=dtype, low=low, high=high)
    assert tensor.shape == size
    assert torch.all(low <= tensor)
    assert torch.all(tensor < high)
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_randn_tensor(size: Sequence[int] = (4, 4), dtype: torch.dtype = torch.float32) -> None:
    """Test randn_tensor for shape and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.randn_tensor(size=size, dtype=dtype)
    assert tensor.shape == size
    if tensor.numel() > 1:
        assert len(torch.unique(tensor)) > 1


def test_randperm(n: int = 5, dtype: torch.dtype = torch.int64) -> None:
    """Test randperm for length, uniqueness, and value differences."""
    rng = RandomGenerator(seed=42)
    tensor = rng.randperm(n=n, dtype=dtype)
    assert len(tensor) == n
    assert len(tensor.unique()) == n
    assert (tensor.unique() == torch.arange(n, dtype=dtype)).all()
