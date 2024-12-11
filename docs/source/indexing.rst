## indexing rules

the data objects can be indexed, similiar to numpy arrays or tensors


1. as in numpy, an index is always a tuple of indices. if not, is is converted to a tuple. the elements the tuple are matched to the dimensions of the data left to right.
   example:
	data[0,1] is the same as data[(0,1)], as the index is converted to a tuple
        data[(0,1),] is not the same as  data[(0,1)], but data[((0,1),)]
2. all indexing operations are perfomred on the fields of a dataclass as the sequence of 
    broadcasting to the dataclasses shape -> indexing -> replacing stride 0 dimensions by singletons as good as possible.
    this means each dataobject can be indexed as if all fields had the shape returend by dataobject.shape, even if the fields
    are only broadcastable to this shape (example: acqidx are singleton in k0)

3. two data objects with the same shape property indexed by the same indexer will result in the same shape property.
     this explicitly does not mean that all fields of the data class have the same shape after indexing. 
     only that the fields can be broadcasted to the common shape

4. slice indexing
	indexing with a slice object or colon synax, e.g. 0:3 will select a slice of the data
	it does a view along that dimension if the stride is positive.
	a[slice1,slice2]=a[slice1,:][:,slice2]=a[:,slice1][slice1,:]
    as the underlying torch.tensor does not support negative strides, a copy will be performed for negative strides.
    this is differnet from numpy which always returns a view for slices and from torch, which errors with negative strides.

5. integer indexing
	integer indexing is the same as indexing with slice index:index+1.
	thus, it does not remove dimensions. this is different than numpy or pytorch.
	use squeeze to remove remaining singleton dimensions if requiered.
	integer indexing always results in a view.

6. advanced indexing with sequences of ints
	if a single indexer is a sequence of ints, example data[:,(0,5)] 
	for each value in the sequence indexing will be performed and the result will be conatenated along the indexing dimension.
	thus, data[:,(0,5)].shape[1]==2.

7. vectorized indexing
	using more than one integer sequence as index will is done by looping over the values in the sequences, performing
	indexing and stacking along a new dimension at position 0.
	example: data[(0,5),:,(2,3)] will result in the same as if one would do stack([data[0,:,2],data[5,:,3],dim=0).
	perticaulary, at the position of the indexed dimensions a singelton dimenion will remain.  	(different than numpy)
	this rulke applies irregedless of wheatrher the indexed dimensions are next to each other or not (different than numpy)

8. indexing with bolean mask
        the boolean mask will first be broadcasted to the shape of the indexed object.
	thus, even though acqidx always are singelton in k0, kdata.header.acqinfo.idx.slice=1 can be used to index kdata.
	the result is the same as indexing with [mask.nonzero(as_tuple=True)[k] if mask.shape[k]!=1 else slice(None) for k in range(mask.ndim)]
	in particular, indexing with a single non singleton dimension will not introduce new axes nor change the shape of the data in the singleton axes.
	but indexing with more than one non-singleton dimension will introduce a new zeroth dimension and result in two singleton dimensins in the data.
	
	
	