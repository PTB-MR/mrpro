"""PyTest fixtures for signal models."""

import pytest
import torch
from tests import RandomGenerator

# Shape combinations for signal models
SHAPE_VARIATIONS_SIGNAL_MODELS = pytest.mark.parametrize(
    ('parameter_shape', 'contrast_dim_shape', 'signal_shape'),
    [
        ((1, 1, 10, 20, 30), (5,), (5, 1, 1, 10, 20, 30)),  # single map with different inversion times
        ((1, 1, 10, 20, 30), (5, 1), (5, 1, 1, 10, 20, 30)),
        ((4, 1, 1, 10, 20, 30), (5, 1), (5, 4, 1, 1, 10, 20, 30)),  # multiple maps along additional batch dimension
        ((4, 1, 1, 10, 20, 30), (5,), (5, 4, 1, 1, 10, 20, 30)),
        ((4, 1, 1, 10, 20, 30), (5, 4), (5, 4, 1, 1, 10, 20, 30)),
        ((3, 1, 10, 20, 30), (5,), (5, 3, 1, 10, 20, 30)),  # multiple maps along other dimension
        ((3, 1, 10, 20, 30), (5, 1), (5, 3, 1, 10, 20, 30)),
        ((3, 1, 10, 20, 30), (5, 3), (5, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5,), (5, 4, 3, 1, 10, 20, 30)),  # multiple maps along other and batch dimension
        ((4, 3, 1, 10, 20, 30), (5, 4), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 4, 1), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 1, 3), (5, 4, 3, 1, 10, 20, 30)),
        ((4, 3, 1, 10, 20, 30), (5, 4, 3), (5, 4, 3, 1, 10, 20, 30)),
        ((1,), (5,), (5, 1)),  # single voxel
        ((4, 3, 1), (5, 4, 3), (5, 4, 3, 1)),
    ],
)


def create_parameter_tensor_tuples(parameter_shape=(10, 5, 100, 100, 100), number_of_tensors=2):
    """Create tuples of tensors as input to operators."""
    random_generator = RandomGenerator(seed=0)
    parameter_tensors = random_generator.float32_tensor(size=(number_of_tensors, *parameter_shape), low=1e-10)
    return torch.unbind(parameter_tensors)
