Indexing Rules
==============

Data objects can be indexed similarly to NumPy arrays or tensors.

1. **Indexing as tuples**

   - As in NumPy, an index is always a tuple of indices. If it is not, it is converted to a tuple. The elements of the tuple are matched to the dimensions of the data from left to right.

   - Example::

       data[0, 1] == data[(0, 1)]  # The index is converted to a tuple
       data[(0, 1), ] != data[(0, 1)] 
	   data[(0, 1), ] == data[((0, 1), )]

2. **Broadcasting**

   - Indexing operations are performed on the fields of a dataclass in the following sequence:

     1. Broadcasting to the dataclass's shape.
     2. Indexing.
     3. Replacing stride-0 dimensions with singletons wherever possible.

   - This means each data object can be indexed as if all fields had the shape returned by ``dataobject.shape``, even if the fields are only broadcastable to this shape.

   - Example: ``acqidx`` is singleton in ``k0``. But the parent kdata object can be indexed along k0.

3. **Consistency of shapes after indexing**

   - Two data objects with the same shape property, indexed by the same indexer, will result in the same shape property.
   - This does *not* mean that all fields of the data class have the same shape after indexing, but rather that the fields can be broadcasted to the common shape.

4. **Slice indexing**

   - Indexing with a slice object or colon syntax (e.g., ``0:3``) selects a slice of the data.
   - A view along the dimension is returned if the stride is positive.

   - Example::

       a[slice1, slice2] == a[slice1, :][:, slice2] == a[:, slice1][slice1, :]

   - For negative strides, a copy is performed because the underlying ``torch.Tensor`` does not support negative strides. This differs from NumPy, which always returns a view, and PyTorch, which raises an error for negative strides.

5. **Integer indexing**

   - Integer indexing is equivalent to slicing with ``index:index+1``, so it does not remove dimensions.
   - This differs from NumPy and PyTorch. Use ``squeeze`` to remove remaining singleton dimensions if needed.
   - Integer indexing always results in a view.

6. **Advanced indexing with sequences of integers**

   - If a single indexer is a sequence of integers (e.g., ``data[:, (0, 5)]``), indexing is performed for each value in the sequence, and the results are concatenated along the indexing dimension.

   - Example::

       data[:, (0, 5)].shape[1] == 2

7. **Vectorized indexing**

   - Using more than one sequence of integers as indices results in vectorized indexing by looping over the values in the sequences, performing indexing, and stacking along a new dimension at position 0.

   - Example::

       data[(0, 5), :, (2, 3)] == stack([data[0, :, 2], data[5, :, 3]], dim=0)

   - At the position of the indexed dimensions, a singleton dimension will remain (unlike NumPy).
   - This rule applies regardless of whether the indexed dimensions are adjacent (unlike NumPy).

8. **Indexing with a boolean mask**

   - The boolean mask is first broadcasted to the shape of the indexed object.

   - Example::

       kdata.header.acqinfo.idx.slice == 1  # Can be used to index `kdata`

   - The result is the same as::

       [mask.nonzero(as_tuple=True)[k] if mask.shape[k] != 1 else slice(None) for k in range(mask.ndim)]

   - Indexing with a single non-singleton dimension will not introduce new axes or change the shape of the singleton dimensions.
   - Indexing with a mask with more than one non-singleton dimension introduces a new zeroth dimension and results in singleton dimensions in the data.
  Example: 
        data.shape=(5,4,3) 
        mask.shape=(5,1,3)
        data[mask].shape==N,1,4,1 
        with N determined by the number of True values o the mask.

9. ellipses and slice(None)
     an indexing expression can contain a single ellipsis, which will be expanded to [slice(None)]*(data.ndim-sum(index_dims)). Here, index_dims in 1 for each slice and integer indices and mask.ndim for boolean indices.

10. None
 using None in an index is not supported. Use the rearrange pattern to introduce new axes.
