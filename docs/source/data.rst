Data Structures, Dimensions, and Indexing
=================================================

In MRtwo, we use `torch.Tensors` and "Dataclasses" to store and manage data. Dataclasses serve as containers for different tensors, metadata, and other dataclasses, ensuring a consistent and structured approach to data handling.

For example `mr2.data.KData` contains complex raw data, trajectory information, and headers.

Dimensions and Broadcasting Rules
---------------------------------
MRtwo follows a consistent convention of using at least 5-dimensional tensors for data representation:

- K-space data: `(*other, coils, k2, k1, k0)`
- Real space (image) data: `(*other, coils, z, y, x)`

Here, `*other` represents additional dimensions, such slice positions, repetitions or cardiac phases. Singleton dimensions are enforced for quantities that do not vary along a specific axis, ensuring they are broadcastable.

For example, a 2D Cartesian trajectory might have:
- kx field: `(other, coils=1, k2=1, k1=1, k0)`
- ky field: `(other, coils=1, k2=1, k1, k0=1)`

Calling `.shape` on the KTrajectory dataclass returns `(other, 1, 1, k1, k0)`.
Neither kx nor ky have this shape, but both can be broadcasted to this shape.

Indexing Conventions
--------------------
Dataclasses support advanced indexing. It follows numpy-like rules with two key exceptions:
1. Indexing never removes dimensions.
2. New dimensions are added at the beginning.

The index can contain slices, integers, boolean masks, sequences of integers, None, and Ellipsis:

- **Slices**: Behave like in numpy, always return a view. Negative step sizes are not supported.
- **Integers**: Behave like slicing with `index:index+1`, always return a view.
- **Boolean masks**: Singleton dimensions in the mask are interpreted as full slices. If the mask has more than one non-singleton dimension, a new dimension is added at the beginning.
- **Sequences of integers**: Result in concatenation along the indexed dimension. If more than one sequence is used, a new dimension is added at the beginning.
- **Integer tensors**: If a single indexing tensor is used, the indexed dimension will be replaced by the last dimension of the indexing tensor. Other dimensions of the indexing tensor are added at the beginning of the result. If multiple indexing tensors are used, the indexed dimension will be replaced by a singleton dimension and all new dimensions are added at the beginning of the result.
- **None**: Can only be used to add new axes at the beginning.
- **Ellipsis**: Expanded to `slice(None)` for all axes not indexed.

The indexing is applied recursively to all tensors in the dataclass, ensuring that the dimensions are consistent across all tensors.
Indexing is applied as if all tensors where broadcasted to the `.shape` of the dataclass the indexing is applied to.
So, for example, indexing a `KData` object in the `coils` dimension results in the same trajecktory as before, as
trajectories always have coils=1. The behavior is as if the trajectory is broadcasted along the coil dimensions, then indexing is applied, then the boadcasted dimensions is reduced back to singleton.


SpatialDimension
----------------
The `mr2.data.SpatialDimension` class represents spatial positions as a named tuple of z, y, and x values. It supports basic math operations and indexing when the values are tensors. This class is used, for example, to store position information in headers.

Rotation
--------
The `mr2.data.Rotation` class handles orientations and rotations. It supports conversion between different representations (e.g., Euler angles, quaternions, rotation matrices) and can be applied to tensors and `SpatialDimension` objects.

Units
-----
All values in MRtwo are in SI units. For example, spatial dimensions are in meters, time is in seconds, and angles are in radians.
This ensures consistency but might require conversion when interfacing with other software or hardware that uses different units,
such as milliseconds for echo times or degrees for flip angles.

Operators
---------
The operators in `mr2.operators` can be applied to one or multiple `torch.Tensor` and return tuples of `torch.Tensor` (even if only a single tensor is returned).
The convention regarding dimensions is the same as for dataclasses: if a new dimension is added, it is added in front (example: time points in signal models) and the dimensions are
in the order `(*other, coils, k2, k1, k0)` for k-space data and `(*other, coil, z, y, x)` for real space data. The operators are always batched, so they can be applied to multiple data points at once.

Examples
--------
Below are practical examples demonstrating the use of dataclasses, indexing, and rotations in MRtwo:

.. code-block:: python

    import mr2

    # Example: Indexing a broadcasted tensor
    kdata = mr2.data.KData.from_file('example.mrd')
    print(kdata.shape)  # Example shape: (4, 8, 64, 64, 128)
    print(kdata.header.acq_info.idx.k1.shape)  # Example shape: (4, 1, 64, 64, 1)

    kdata_center = kdata[... 16:-16, 16:-16,16:-16]
    print(kdata_center.shape)  # Example shape: (4, 8, 32, 32, 96)
    print(kdata.header.acq_info.idx.k1.shape)  # Example shape: (4, 1, 32, 32, 1)

    # Example: Creating and using SpatialDimension and Rotation
    position = mr2.data.SpatialDimension(z=3, y=2, x=1)
    rotation = mr2.data.Rotation.from_euler("xyz", (0,0,torch.pi))
    rotated = rotation(position)
