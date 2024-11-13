"""Random generator."""

from collections.abc import Sequence

import torch


class RandomGenerator:
    """Generate random numbers for testing purposes. Uses a fixed seed to
    ensure reproducibility.

    provides:
        scalar uniform random numbers:
            int8, int16, int32, int64, uint8, uint16, uint32, uint64
            float32, float64
            complex64, complex128
        tensor of uniform random numbers:
            int8_tensor, int16_tensor, int32_tensor, int64_tensor, uint8_tensor
            Note: uint16, uint32 and uint64 are not yet supported by pytorch
            float32_tensor, float64_tensor
            complex64_tensor, complex128_tensor
        tuple of uniform random numbers:
            int8_tuple, int16_tuple, int32_tuple, int64_tuple
            uint8_tuple, uint16_tuple, uint32_tuple, uint64_tuple
            float32_tuple, float64_tuple
            complex64_tuple, complex128_tuple
    """

    def __init__(self, seed):
        """Initialize with a fixed seed."""
        self.generator = torch.Generator().manual_seed(seed)

    @staticmethod
    def _clip_bounds(low, high, lowest, highest):
        """Clips the bounds (low, high) to the given range (lowest, highest)."""
        if low > high:
            raise ValueError('low should be lower than high')
        low = max(low, lowest)
        high = max(low, min(high, highest))
        return low, high

    @staticmethod
    def _dtype_bounds(dtype):
        """Returns the bounds of a given dtype."""
        info: torch.finfo | torch.iinfo
        if dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            # Integer types
            info = torch.iinfo(dtype)
        elif dtype is None:
            # Defaults to pytorch default dtype
            info = torch.finfo()
        else:
            # Float types
            info = torch.finfo(dtype)
        return (info.min, info.max)

    def _randint(self, size, low, high, dtype=torch.int64) -> torch.Tensor:
        """Generate uniform random integers in [low, high) with given dtype."""
        low, high = self._clip_bounds(low, high, *self._dtype_bounds(dtype))
        return torch.randint(low, high, size, generator=self.generator, dtype=dtype)

    def _rand(self, size, low, high, dtype=torch.float32) -> torch.Tensor:
        """Generate uniform random floats in [low, high) with given dtype."""
        low, high = self._clip_bounds(low, high, *self._dtype_bounds(dtype))
        return (torch.rand(size, generator=self.generator, dtype=dtype) * (high - low)) + low

    def float32_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate float32 tensor of given size in [low, high)."""
        return self._rand(size, low, high, torch.float32)

    def float64_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate float64 tensor of given size in [low, high)."""
        return self._rand(size, low, high, torch.float64)

    def complex64_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate complex64 tensor of given size in [low, high)."""
        if low < 0:
            raise ValueError('low/high refer to the amplitude and must be positive')
        amp = self.float32_tensor(size, low, high)
        phase = self.float32_tensor(size, -torch.pi, torch.pi)
        return (amp * torch.exp(1j * phase)).to(dtype=torch.complex64)

    def complex128_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate complex128 tensor of given size in [low, high)."""
        if low < 0:
            raise ValueError('low/high refer to the amplitude and must be positive')
        amp = self.float64_tensor(size, low, high)
        phase = self.float64_tensor(size, -torch.pi, torch.pi)
        return (amp * torch.exp(1j * phase)).to(dtype=torch.complex128)

    def int8_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 7, high: int = 1 << 7) -> torch.Tensor:
        """Generate int8 tensor of given size in [low, high)."""
        return self._randint(size, low, high, dtype=torch.int8)

    def int16_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 15, high: int = 1 << 15) -> torch.Tensor:
        """Generate int16 tensor of given size in [low, high)."""
        return self._randint(size, low, high, dtype=torch.int16)

    def int32_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 31, high: int = 1 << 31) -> torch.Tensor:
        """Generate int32 tensor of given size in [low, high)."""
        return self._randint(size, low, high, dtype=torch.int32)

    def int64_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 63, high: int = 1 << 63) -> torch.Tensor:
        """Generate int64 tensor of given size in [low, high)."""
        return self._randint(size, low, high, dtype=torch.int64)

    # There is no uint32 in pytorch yet
    # def uint32_tensor(self, size: Sequence[int] = (1,), low: int = 0, high: int = 1 << 32):
    #     return self._randint(size, low, high, dtype=torch.int32) # noqa: ERA001

    # There is no uint64 in pytorch yet
    # def uint64_tensor(self, size: Sequence[int] = (1,), low: int = 0, high: int = 1 << 64):
    #    return self._randint(size, low, high, dtype=torch.uint64) # noqa: ERA001

    def uint8_tensor(self, size: Sequence[int] | int = (1,), low: int = 0, high: int = 1 << 8) -> torch.Tensor:
        """Generate uint8 tensor of given size in [low, high)."""
        return self._randint(size, low, high, dtype=torch.uint8)

    def bool(self) -> bool:
        """Generate a random boolean value."""
        return self.uint8(0, 1) == 1

    def float32(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a float32 in [low, high)."""
        return self.float32_tensor((1,), low, high).item()

    def float64(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a float64 in [low, high)."""
        return self.float64_tensor((1,), low, high).item()

    def complex64(self, low: float = 0, high: float = 1.0) -> complex:
        """Generate a complex64 in [low, high)."""
        return self.complex64_tensor((1,), low, high).item()

    def complex128(self, low: float = 0, high: float = 1.0) -> complex:
        """Generate a complex128 in [low, high)."""
        return self.complex128_tensor((1,), low, high).item()

    def uint8(self, low: int = 0, high: int = 1 << 8) -> int:
        """Generate a uint8 in [low, high)."""
        return int(self.uint8_tensor((1,), low, high).item())

    def uint16(self, low: int = 0, high: int = 1 << 16) -> int:
        """Generate a uint16 in [low, high)."""
        if low < 0 or high > 1 << 16:
            raise ValueError('Low must be positive and high must be <= 2^16')
        # using int32 as it is the smallest that can hold 2^16 (no uint32 in pytorch)
        return int(self.int32_tensor((1,), low, high).item())

    def uint32(self, low: int = 0, high: int = 1 << 32) -> int:
        """Generate a uint32 in [low, high)."""
        if low < 0 or high > 1 << 32:
            raise ValueError('Low must be positive and high must be <= 2^32')
        # using int64 as it is the smallest that can hold 2^32 (no uint64 in pytorch)
        return int(self.int64_tensor((1,), low, high).item())

    def int8(self, low: int = -1 << 7, high: int = 1 << 7 - 1) -> int:
        """Generate an int8 in [low, high)."""
        return int(self.int8_tensor((1,), low, high).item())

    def int16(self, low: int = -1 << 15, high: int = 1 << 15 - 1) -> int:
        """Generate an int16 in [low, high)."""
        return int(self.int16_tensor((1,), low, high).item())

    def int32(self, low: int = -1 << 31, high: int = 1 << 31 - 1) -> int:
        """Generate an int32 in [low, high)."""
        return int(self.int32_tensor((1,), low, high).item())

    def int64(self, low: int = -1 << 63, high: int = 1 << 63 - 1) -> int:
        """Generate an int64 in [low, high)."""
        return int(self.int64_tensor((1,), low, high).item())

    def uint64(self, low: int = 0, high: int = 1 << 64) -> int:
        """Generate a uint64 in [low, high)."""
        if low < 0 or high > 1 << 64:
            raise ValueError('Low must be positive and high must be <= 2^64')
        # no uint64 in pytorch. int64 would not be able to produce 2^64,
        # so we need to shift the values from [-2^63, 2^63) to [0, 2^64)
        range_ = high - low
        new_low = -1 << 63
        new_high = new_low + range_
        value = self.int64(new_low, new_high) - new_low + low
        return value

    def float32_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[float, ...]:
        """Generate a tuple of float32 of given size in [low, high)."""
        return tuple(self.float32_tensor((size,), low, high))

    def float64_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[float, ...]:
        """Generate a tuple of float64 of given size in [low, high)."""
        return tuple(self.float64_tensor((size,), low, high))

    def complex64_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[complex, ...]:
        """Generate a tuple of complex64 of given size in [low, high)."""
        return tuple(self.complex64_tensor((size,), low, high))

    def complex128_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[complex, ...]:
        """Generate a tuple of complex128 of given size in [low, high)."""
        return tuple(self.complex128_tensor((size,), low, high))

    def uint8_tuple(self, size: int, low: int = 0, high: int = 1 << 8) -> tuple[int, ...]:
        """Generate a tuple of uint8 of given size in [low, high)."""
        return tuple(self.uint8_tensor((size,), low, high))

    def uint16_tuple(self, size: int, low: int = 0, high: int = 1 << 16) -> tuple[int, ...]:
        """Generate a tuple of uint16 of given size in [low, high)."""
        return tuple([self.uint16(low, high) for _ in range(size)])

    def uint32_tuple(self, size: int, low: int = 0, high: int = 1 << 32) -> tuple[int, ...]:
        """Generate a tuple of uint32 of given size in [low, high)."""
        return tuple([self.uint32(low, high) for _ in range(size)])

    def uint64_tuple(self, size: int, low: int = 0, high: int = 1 << 64) -> tuple[int, ...]:
        """Generate a tuple of uint64 of given size in [low, high)."""
        return tuple([self.uint64(low, high) for _ in range(size)])

    def int8_tuple(self, size: int, low: int = -1 << 7, high: int = 1 << 7) -> tuple[int, ...]:
        """Generate a tuple of int8 of given size in [low, high)."""
        return tuple(self.int8_tensor((size,), low, high))

    def int16_tuple(self, size: int, low: int = -1 << 15, high: int = 1 << 15) -> tuple[int, ...]:
        """Generate a tuple of int16 of given size in [low, high)."""
        return tuple(self.int16_tensor((size,), low, high))

    def int32_tuple(self, size: int, low: int = -1 << 31, high: int = 1 << 31) -> tuple[int, ...]:
        """Generate a tuple of int32 of given size in [low, high)."""
        return tuple(self.int32_tensor((size,), low, high))

    def int64_tuple(self, size: int, low: int = -1 << 63, high: int = 1 << 63) -> tuple[int, ...]:
        """Generate a tuple of int64 of given size in [low, high)."""
        return tuple(self.int64_tensor((size,), low, high))

    def ascii(self, size: int) -> str:
        """Generate a random ASCII string of given size."""
        return ''.join([chr(self.uint8(32, 127)) for _ in range(size)])

    def rand_like(self, x: torch.Tensor, low=0.0, high=1.0) -> torch.Tensor:
        """Generate tensor like x with uniform random numbers in [low, high)."""
        return self.rand_tensor(x.shape, x.dtype, low=low, high=high)

    def rand_tensor(self, shape: Sequence[int], dtype: torch.dtype, low: float, high: float) -> torch.Tensor:
        """Generate tensor of given shape and dtype in [low, high)."""
        if dtype.is_complex:
            tensor = self.complex64_tensor(shape, low, high).to(dtype=dtype)
        elif dtype.is_floating_point:
            tensor = self._rand(shape, low, high, dtype)
        elif dtype == torch.bool:
            tensor = self.int32_tensor(shape, low=0, high=1) > 0
        else:
            tensor = self._randint(shape, low, high, dtype)
        return tensor

    def randperm(self, n, *, dtype=torch.int64) -> torch.Tensor:
        """Generate random permutation of integers from 0 to n-1."""
        return torch.randperm(n, generator=self.generator, dtype=dtype)
