from itertools import product

import pytest
import torch
from mrpro.operators.models import MOLLI
from mrpro.operators.models import WASABI
from mrpro.operators.models import WASABITI
from mrpro.operators.models import InversionRecovery
from mrpro.operators.models import SaturationRecovery
from tests import RandomGenerator

# Signal model, number of tensor arguments in forward (i.e. quantitative parameters) and number of tensor arguments
# in init (e.g. inversion time or CEST offsets)
MODELS_AND_N_INPUT_PARAMETERS = [
    (MOLLI, 3, 1),
    (InversionRecovery, 2, 1),
    (SaturationRecovery, 2, 1),
    (WASABI, 4, 1),
    (WASABITI, 3, 2),
]

# Shape combinations
SHAPE_VARIATIONS_SIGNAL_MODELS = [
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
]

# Combine models and shapes
SHAPE_TESTS = pytest.mark.parametrize(
    ('model', 'n_input_tensors', 'n_init_parameters', 'parameter_shape', 'contrast_dim_shape', 'signal_shape'),
    [
        (*model_shape[0], *model_shape[1])
        for model_shape in product(MODELS_AND_N_INPUT_PARAMETERS, SHAPE_VARIATIONS_SIGNAL_MODELS)
    ],
)


def create_parameter_tensors(parameter_shape=(10, 5, 100, 100, 100), number_of_tensors=2):
    """Create list of tensors as input to signal models."""
    random_generator = RandomGenerator(seed=0)
    parameter_tensors = random_generator.float32_tensor(size=(number_of_tensors, *parameter_shape), low=1e-10)
    return torch.unbind(parameter_tensors)


@SHAPE_TESTS
def test_model_shape(model, n_input_tensors, n_init_parameters, parameter_shape, contrast_dim_shape, signal_shape):
    """Test correct signal shapes."""
    input_parameters = create_parameter_tensors(contrast_dim_shape, number_of_tensors=n_init_parameters)
    model_op = model(*input_parameters)
    parameter_tensors = create_parameter_tensors(parameter_shape, number_of_tensors=n_input_tensors)
    (signal,) = model_op.forward(*parameter_tensors)
    assert signal.shape == signal_shape
