"""PyTest fixtures for signal models."""

import pytest

# Shape combinations for signal models
SHAPE_VARIATIONS_SIGNAL_MODELS = pytest.mark.parametrize(
    ('parameter_shape', 'contrast_dim_shape', 'signal_shape'),
    [
        ((1, 1, 10, 20, 30), (5,), (5, 1, 1, 10, 20, 30)),  # single map with different contrast times
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
        ((4, 3, 1, 10, 20, 30), (5, 4, 3, 1, 10, 20, 30), (5, 4, 3, 1, 10, 20, 30)),  # different value for each voxel
        ((1,), (5,), (5, 1)),  # single voxel
        ((4, 3, 1), (5, 4, 3), (5, 4, 3, 1)),
    ],
    ids=[
        'single_map_diff_contrast_times',
        'single_map_diff_contrast_times_2',
        'multiple_maps_additional_batch_dim',
        'multiple_maps_additional_batch_dim_2',
        'multiple_maps_additional_batch_dim_3',
        'multiple_maps_other_dim',
        'multiple_maps_other_dim_2',
        'multiple_maps_other_dim_3',
        'multiple_maps_other_and_batch_dim',
        'multiple_maps_other_and_batch_dim_2',
        'multiple_maps_other_and_batch_dim_3',
        'multiple_maps_other_and_batch_dim_4',
        'multiple_maps_other_and_batch_dim_5',
        'different_value_each_voxel',
        'single_voxel',
        'multiple_voxels',
    ],
)
