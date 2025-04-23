"""Random generator."""

from collections.abc import Sequence
from math import ceil, floor

import torch


def check_bounds(low: float | int | torch.Tensor, high: float | int | torch.Tensor, dtype: torch.dtype | None) -> None:
    """Clip the bounds to a range matching the dtype.

    Parameters
    ----------
    low
        Lower bound.
    high
        Upper bound.
    dtype
        Data type, used to find allowed range.
    """
    info: torch.finfo | torch.iinfo
    if dtype is None:
        info = torch.finfo()
        minval, maxval = info.min, info.max
    elif dtype.is_floating_point:
        info = torch.finfo(dtype)
        minval, maxval = info.min, info.max
    else:
        info = torch.iinfo(dtype)
        minval = info.min
        if dtype in (torch.int64, torch.uint64):
            maxval = info.max  # https://github.com/pytorch/pytorch/issues/81446
        else:
            maxval = info.max + 1
    if low > high:
        raise ValueError('low should be lower than high')
    if low < minval or high > maxval:
        raise ValueError(f'low/high should be in the range of {info.min} and {info.max} for {dtype}')


class RandomGenerator:
    """Generate random numbers for various purposes.

    Uses a fixed seed to ensure reproducibility.

    Provides:
        - Scalar uniform random numbers:
            int8, int16, int32, int64, uint8, uint16, uint32, uint64,
            float32, float64, complex64, complex128
        - Tensor of uniform random numbers:
            int8_tensor, int16_tensor, int32_tensor, int64_tensor, uint8_tensor,
            float32_tensor, float64_tensor, complex64_tensor, complex128_tensor
            (Note: uint16, uint32, uint64 tensors are not yet supported by PyTorch)
        - Tuple of uniform random numbers:
            int8_tuple, int16_tuple, int32_tuple, int64_tuple,
            uint8_tuple, uint16_tuple, uint32_tuple, uint64_tuple,
            float32_tuple, float64_tuple, complex64_tuple, complex128_tuple
    """

    def __init__(self, seed: int):
        """Initialize the random generator with a fixed seed."""
        self.generator = torch.Generator().manual_seed(seed)

    def _randint(
        self, size: Sequence[int] | int, low: int, high: int, dtype: torch.dtype = torch.int64
    ) -> torch.Tensor:
        """Generate uniform random integers in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).
        dtype
            Data type of the output tensor.

        Returns
        -------
            Tensor of random integers.
        """
        check_bounds(low, high, dtype)
        size_ = (size,) if isinstance(size, int) else size
        return torch.randint(low, high, size_, generator=self.generator, dtype=dtype)

    def _rand(
        self,
        size: Sequence[int] | int,
        low: float | torch.Tensor,
        high: float | torch.Tensor,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Generate uniform random floats in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound.
        high
            Upper bound.
        dtype
            Data type of the output tensor.

        Returns
        -------
            Tensor of random floats.
        """
        check_bounds(low, high, dtype)
        return (torch.rand(size, generator=self.generator, dtype=dtype) * (high - low)) + low

    def float32_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate a float32 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Tensor of float32 random numbers.
        """
        return self._rand(size, low, high, torch.float32)

    def float64_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate a float64 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Tensor of float64 random numbers.
        """
        return self._rand(size, low, high, torch.float64)

    def complex64_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate a complex64 tensor with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Tensor of complex64 random numbers.
        """
        if low < 0:
            raise ValueError('low/high refer to the amplitude and must be positive')
        amp = self.float32_tensor(size, low, high)
        phase = self.float32_tensor(size, -torch.pi, torch.pi)
        return (amp * torch.exp(1j * phase)).to(dtype=torch.complex64)

    def complex128_tensor(self, size: Sequence[int] | int = (1,), low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate a complex128 tensor with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Tensor of complex128 random numbers.
        """
        if low < 0:
            raise ValueError('low/high refer to the amplitude and must be positive')
        amp = self.float64_tensor(size, low, high)
        phase = self.float64_tensor(size, -torch.pi, torch.pi)
        return (amp * torch.exp(1j * phase)).to(dtype=torch.complex128)

    def int8_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 7, high: int = 1 << 7) -> torch.Tensor:
        """Generate an int8 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tensor of int8 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.int8)

    def int16_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 15, high: int = 1 << 15) -> torch.Tensor:
        """Generate an int16 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tensor of int16 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.int16)

    def int32_tensor(self, size: Sequence[int] | int = (1,), low: int = -1 << 31, high: int = 1 << 31) -> torch.Tensor:
        """Generate an int32 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tensor of int32 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.int32)

    def int64_tensor(
        self, size: Sequence[int] | int = (1,), low: int = -1 << 63, high: int = (1 << 63) - 1
    ) -> torch.Tensor:
        """Generate an int64 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).
            Maximum value is (1 << 63) - 1 due to https://github.com/pytorch/pytorch/issues/81446

        Returns
        -------
            Tensor of int64 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.int64)

    def uint8_tensor(self, size: Sequence[int] | int = (1,), low: int = 0, high: int = 1 << 8) -> torch.Tensor:
        """Generate a uint8 tensor with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tensor of uint8 random numbers.
        """
        return self._randint(size, low, high, dtype=torch.uint8)

    def bool_tensor(self, size: Sequence[int] | int = (1,)) -> torch.Tensor:
        """Generate boolean tensor of given size.

        Parameters
        ----------
        size
            Shape of the output tensor.
        """
        return self.uint8_tensor(size, low=0, high=2).bool()

    def bool(self) -> bool:
        """Generate a random boolean value.

        Returns
        -------
            Random boolean.
        """
        return self.uint8(0, 1) == 1

    def float32(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a float32 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Random float32 number.
        """
        return self.float32_tensor((1,), low, high).item()

    def float64(self, low: float = 0.0, high: float = 1.0) -> float:
        """Generate a float64 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Random float64 number.
        """
        return self.float64_tensor((1,), low, high).item()

    def complex64(self, low: float = 0, high: float = 1.0) -> complex:
        """Generate a complex64 scalar with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Random complex64 number.
        """
        return self.complex64_tensor((1,), low, high).item()

    def complex128(self, low: float = 0, high: float = 1.0) -> complex:
        """Generate a complex128 scalar with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Random complex128 number.
        """
        return self.complex128_tensor((1,), low, high).item()

    def uint8(self, low: int = 0, high: int = (1 << 8) - 1) -> int:
        """Generate a uint8 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint8 number.
        """
        return int(self.uint8_tensor((1,), low, high).item())

    def uint16(self, low: int = 0, high: int = 1 << 16) -> int:
        """Generate a uint16 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint16 number.
        """
        if low < 0 or high > 1 << 16:
            raise ValueError('Low must be positive and high must be <= 2^16')
        return int(self.int32_tensor((1,), low, high).item())

    def uint32(self, low: int = 0, high: int = 1 << 32) -> int:
        """Generate a uint32 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint32 number.
        """
        if low < 0 or high > 1 << 32:
            raise ValueError('Low must be positive and high must be <= 2^32')
        return int(self.int64_tensor((1,), low, high).item())

    def int8(self, low: int = -1 << 7, high: int = 1 << 7) -> int:
        """Generate an int8 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int8 number.
        """
        return int(self.int8_tensor((1,), low, high).item())

    def int16(self, low: int = -1 << 15, high: int = 1 << 15) -> int:
        """Generate an int16 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int16 number.
        """
        return int(self.int16_tensor((1,), low, high).item())

    def int32(self, low: int = -1 << 31, high: int = 1 << 31) -> int:
        """Generate an int32 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int32 number.
        """
        return int(self.int32_tensor((1,), low, high).item())

    def int64(self, low: int = -1 << 63, high: int = (1 << 63) - 1) -> int:
        """Generate an int64 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random int64 number.
        """
        return int(self.int64_tensor((1,), low, high).item())

    def uint64(self, low: int = 0, high: int = (1 << 64) - 1) -> int:
        """Generate a uint64 scalar with uniform distribution in [low, high).

        Parameters
        ----------
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Random uint64 number.
        """
        if low < 0 or high > 1 << 64:
            raise ValueError('Low must be positive and high must be <= 2^64')
        range_ = high - low
        new_low = -1 << 63
        new_high = new_low + range_
        value = self.int64(new_low, new_high) - new_low + low
        return value

    def float32_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[float, ...]:
        """Generate a tuple of float32 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Tuple of float32 random numbers.
        """
        return tuple(self.float32_tensor((size,), low, high))

    def float64_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[float, ...]:
        """Generate a tuple of float64 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Tuple of float64 random numbers.
        """
        return tuple(self.float64_tensor((size,), low, high))

    def complex64_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[complex, ...]:
        """Generate a tuple of complex64 numbers with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Tuple of complex64 random numbers.
        """
        return tuple(self.complex64_tensor((size,), low, high))

    def complex128_tuple(self, size: int, low: float = 0, high: float = 1) -> tuple[complex, ...]:
        """Generate a tuple of complex128 numbers with uniform amplitude in [low, high).

        The phase is uniformly distributed in [-π, π].

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound for amplitude (must be non-negative).
        high
            Upper bound for amplitude.

        Returns
        -------
            Tuple of complex128 random numbers.
        """
        return tuple(self.complex128_tensor((size,), low, high))

    def uint8_tuple(self, size: int, low: int = 0, high: int = 1 << 8) -> tuple[int, ...]:
        """Generate a tuple of uint8 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint8 random numbers.
        """
        return tuple(self.uint8_tensor((size,), low, high))

    def uint16_tuple(self, size: int, low: int = 0, high: int = 1 << 16) -> tuple[int, ...]:
        """Generate a tuple of uint16 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint16 random numbers.
        """
        return tuple([self.uint16(low, high) for _ in range(size)])

    def uint32_tuple(self, size: int, low: int = 0, high: int = 1 << 32) -> tuple[int, ...]:
        """Generate a tuple of uint32 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint32 random numbers.
        """
        return tuple([self.uint32(low, high) for _ in range(size)])

    def uint64_tuple(self, size: int, low: int = 0, high: int = (1 << 64) - 1) -> tuple[int, ...]:
        """Generate a tuple of uint64 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of uint64 random numbers.
        """
        return tuple([self.uint64(low, high) for _ in range(size)])

    def int8_tuple(self, size: int, low: int = -1 << 7, high: int = 1 << 7) -> tuple[int, ...]:
        """Generate a tuple of int8 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int8 random numbers.
        """
        return tuple(self.int8_tensor((size,), low, high))

    def int16_tuple(self, size: int, low: int = -1 << 15, high: int = 1 << 15) -> tuple[int, ...]:
        """Generate a tuple of int16 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int16 random numbers.
        """
        return tuple(self.int16_tensor((size,), low, high))

    def int32_tuple(self, size: int, low: int = -1 << 31, high: int = 1 << 31) -> tuple[int, ...]:
        """Generate a tuple of int32 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int32 random numbers.
        """
        return tuple(self.int32_tensor((size,), low, high))

    def int64_tuple(self, size: int, low: int = -1 << 63, high: int = (1 << 63) - 1) -> tuple[int, ...]:
        """Generate a tuple of int64 numbers with uniform distribution in [low, high).

        Parameters
        ----------
        size
            Number of elements.
        low
            Lower bound (inclusive).
        high
            Upper bound (exclusive).

        Returns
        -------
            Tuple of int64 random numbers.
        """
        return tuple(self.int64_tensor((size,), low, high))

    def ascii(self, size: int) -> str:
        """Generate a random ASCII string.

        Parameters
        ----------
        size
            Length of the string.

        Returns
        -------
            Random ASCII string.
        """
        return ''.join([chr(self.uint8(32, 127)) for _ in range(size)])

    def rand_like(self, x: torch.Tensor, low: float = 0.0, high: float = 1.0) -> torch.Tensor:
        """Generate a tensor with the same shape and dtype as `x`, filled with uniform random numbers.

        Parameters
        ----------
        x
            Reference tensor.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Random tensor with the same shape and dtype as `x`.
        """
        return self.rand_tensor(x.shape, x.dtype, low=low, high=high)

    def rand_tensor(
        self, size: Sequence[int], dtype: torch.dtype, low: float | int = 0, high: int | float = 1
    ) -> torch.Tensor:
        """Generate a tensor of given shape and dtype with uniform random numbers in [low, high).

        Parameters
        ----------
        size
            Shape of the output tensor.
        dtype
            Data type of the output tensor.
        low
            Lower bound.
        high
            Upper bound.

        Returns
        -------
            Random tensor.
        """
        if dtype.is_complex:
            tensor = self.complex64_tensor(size, low, high).to(dtype=dtype)
        elif dtype.is_floating_point:
            tensor = self._rand(size, low, high, dtype)
        elif dtype == torch.bool:
            tensor = self.int32_tensor(size, low=0, high=1) > 0
        else:
            tensor = self._randint(size, ceil(low), floor(high), dtype)
        return tensor

    def randn_tensor(self, size: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Generate a tensor of given shape and dtype with standard normal distribution.

        Parameters
        ----------
        size
            Shape of the output tensor.
        dtype
            Data type of the output tensor.

        Returns
        -------
            Random tensor with normal distribution.
        """
        return torch.randn(size=size, generator=self.generator, dtype=dtype)

    def randperm(self, n: int, *, dtype: torch.dtype = torch.int64) -> torch.Tensor:
        """Generate a random permutation of integers from 0 to n - 1.

        Parameters
        ----------
        n
            Number of elements.
        dtype
            Data type of the output tensor.

        Returns
        -------
            Tensor containing a random permutation.
        """
        return torch.randperm(n, generator=self.generator, dtype=dtype)

    def gaussian_variable_density_samples(
        self, shape: Sequence[int], low: int, high: int, fwhm: float = float('inf'), always_sample: Sequence[int] = ()
    ) -> torch.Tensor:
        """Generate Gaussian variable density samples.

        Generates indices in [low, high[ with a gaussian weighting.

        Parameters
        ----------
        shape
            Shape of the output tensor. The generated indices are 1D and in the last dimension.
            All other dimensions are batch dimensions.
        low
            Lower bound of the sampling domain.
        high
            Upper bound of the sampling domain.
        fwhm
            Full-width at half-maximum of the Gaussian.
        always_sample
            indices that should always included in the samples.
            For example, `range(-n_center//2, n_center//2)`

        Returns
        -------
            1D tensor of selected indices.
        """
        *n_batch, n_samples = shape
        if n_samples > high - low:
            raise ValueError('n_samples must be <= (high - low)')
        n_random = n_samples - len(always_sample)
        if n_random < 0:
            raise ValueError('more always sampled points requested than total number of samples')
        elif n_random == 0:
            return torch.tensor(always_sample).sort().values.broadcast_to(*n_batch, -1)
        pdf = torch.exp(-torch.tensor(2.0).log() * (2 * torch.arange(low, high) / fwhm) ** 2)
        pdf[[x - low for x in always_sample]] = 0
        if len(shape) > 1:
            pdf = pdf.broadcast_to((*n_batch, -1)).flatten(end_dim=-2)

        idx_rand = pdf.multinomial(n_random, False, generator=self.generator) + low
        if len(shape) > 1:
            idx_rand = idx_rand.unflatten(0, n_batch)
        idx_always = torch.tensor(always_sample).broadcast_to(*n_batch, -1)
        return torch.cat([idx_rand, idx_always], -1).sort().values
