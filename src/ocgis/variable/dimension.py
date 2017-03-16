import numpy as np

from ocgis import constants
from ocgis.base import AbstractNamedObject
from ocgis.constants import DataTypes
from ocgis.util.helpers import get_formatted_slice
from ocgis.vm.mpi import get_global_to_local_slice, MPI_COMM, get_nonempty_ranks


class Dimension(AbstractNamedObject):
    def __init__(self, name, size=None, size_current=None, src_idx=None, dist=False, is_empty=False,
                 source_name=constants.UNINITIALIZED, aliases=None, uid=None):
        if isinstance(src_idx, basestring):
            if src_idx != 'auto' and size is None and size_current is None:
                raise ValueError('Unsized dimensions should not have source indices.')
            if src_idx != 'auto':
                raise ValueError('"src_idx" argument not recognized: {}'.format(src_idx))
        if is_empty and not dist:
            raise ValueError('Undistributed dimensions may not be empty.')

        super(Dimension, self).__init__(name, aliases=aliases, source_name=source_name, uid=uid)

        self.__src_idx__ = None
        self._bounds_global = None
        self._bounds_local = None
        self._size = size
        self._size_current = size_current
        self.dist = dist
        self._is_empty = is_empty

        if not self.is_empty:
            self.set_size(self.size or self.size_current, src_idx=src_idx)
        else:
            self.convert_to_empty()

    def __eq__(self, other):
        ret = True
        skip = ('__src_idx__', '_aliases', '_source_name', '_bounds_local', '_bounds_global')
        for k, v in self.__dict__.items():
            if k in skip:
                continue
            else:
                if v != other.__dict__[k]:
                    ret = False
                    break
        if ret:
            if self._src_idx is None and other._src_idx is not None:
                ret = False
            else:
                if not np.all(self._src_idx == other._src_idx):
                    ret = False
        return ret

    def __getitem__(self, slc):
        # We cannot slice zero length dimensions.
        if len(self) == 0:
            raise IndexError('Zero-length dimensions are not slicable.')

        slc = get_formatted_slice(slc, 1)[0]
        ret = self.copy()

        # Slicing work is done here.
        self.__getitem_main__(ret, slc)

        return ret

    def __len__(self):
        if self.size is None:
            ret = self.size_current
        else:
            ret = self.size
        if ret is None:
            ret = 0
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        template = "{0}(name='{1}', size={2}, size_current={3}, dist={4}, is_empty={5})"
        msg = template.format(self.__class__.__name__, self.name, self.size, self.size_current, self.dist,
                              self.is_empty)
        return msg

    @property
    def bounds_global(self):
        if self._bounds_global is None:
            ret = (0, len(self))
        else:
            ret = self._bounds_global
        return ret

    @bounds_global.setter
    def bounds_global(self, value):
        if value is not None:
            value = tuple(value)
            assert len(value) == 2
        self._bounds_global = value

    @property
    def bounds_local(self):
        if self._bounds_local is None:
            ret = (0, len(self))
        else:
            ret = self._bounds_local
        return ret

    @bounds_local.setter
    def bounds_local(self, value):
        if value is not None:
            value = tuple(value)
            assert len(value) == 2
        self._bounds_local = value

    @property
    def is_empty(self):
        return self._is_empty

    @property
    def is_unlimited(self):
        if self.size is None:
            ret = True
        else:
            ret = False
        return ret

    @property
    def name(self):
        return self._name

    @property
    def size(self):
        return self._size

    @property
    def size_current(self):
        if self._size_current is None:
            ret = self.size
        else:
            ret = self._size_current
        return ret

    @property
    def _src_idx(self):
        return self.__src_idx__

    @_src_idx.setter
    def _src_idx(self, value):
        if value is not None:
            if not isinstance(value, np.ndarray):
                value = np.array(value)
            if len(value) != len(self):
                raise ValueError('Source index length must equal the dimension length.')
        self.__src_idx__ = value

    def convert_to_empty(self):
        self._bounds_local = (0, 0)
        self._src_idx = None
        self._is_empty = True

    def get_distributed_slice(self, slc, comm=None):
        slc = get_formatted_slice(slc, 1)[0]
        if not isinstance(slc, slice):
            raise ValueError('Slice not recognized: {}'.format(slc))

        # None slices return all dimension elements.
        if slc == slice(None):
            ret = self.copy()
        # Use standard slicing for non-distributed dimensions.
        elif not self.dist:
            ret = self[slc]
        else:
            dimension_size = 0
            # Handle empty dimensions. They will have zero size.
            if self.is_empty:
                ret = self.copy()
                ret.convert_to_empty()
            # Handle non-empty dimension slicing.
            else:
                local_slc = get_global_to_local_slice((slc.start, slc.stop), self.bounds_local)
                # Slice does not overlap local bounds. The dimension is now empty with size 0.
                if local_slc is None:
                    ret = self.copy()
                    ret.convert_to_empty()
                # Slice overlaps and do a slice on the dimension using the local slice.
                else:
                    ret = self[slice(*local_slc)]
                    dimension_size = len(ret)

            # Find the length of the distributed dimension and update the global dimension information.
            comm = comm or MPI_COMM
            rank = comm.Get_rank()
            size = comm.Get_size()

            # Sum dimension sizes (empties will be zero). This sum is the new global bounds.
            assert dimension_size >= 0
            dimension_sizes = comm.gather(dimension_size)
            if rank == 0:
                sum_dimension_size = 0
                for ds in dimension_sizes:
                    try:
                        sum_dimension_size += ds
                    except TypeError:
                        pass
                bounds_global = (0, sum_dimension_size)
            else:
                bounds_global = None
            bounds_global = comm.bcast(bounds_global)
            ret.bounds_global = bounds_global

            # Normalize the local bounds on live ranks.
            non_empty_ranks = get_nonempty_ranks(ret, comm=comm)
            non_empty_ranks = comm.bcast(non_empty_ranks)
            if rank == non_empty_ranks[0]:
                adjust = len(ret)
            else:
                adjust = None
            adjust = comm.bcast(adjust, root=non_empty_ranks[0])
            for current_rank in range(size):
                if rank == current_rank:
                    if rank in non_empty_ranks and rank != non_empty_ranks[0]:
                        ret.bounds_local = [b + adjust for b in ret.bounds_local]
                        adjust += len(ret)
                adjust = comm.bcast(adjust, root=current_rank)

        return ret

    def set_size(self, value, src_idx=None):
        if value is not None:
            if isinstance(src_idx, basestring) and src_idx == 'auto':
                src_idx = create_src_idx(0, value, dtype=DataTypes.DIMENSION_SRC_INDEX)
        elif value is None:
            src_idx = None
        else:
            pass

        self._bounds_local = None
        self._size_current = value
        if not self.is_unlimited:
            self._size = value
        self._src_idx = src_idx

        if self.dist:
            # A size definition is required.
            if self.size is None and self.size_current is None:
                msg = 'Distributed dimensions require a size definition using "size" or "size_current".'
                raise ValueError(msg)

    def __getitem_main__(self, ret, slc):
        length_self = len(self)
        try:
            length = len(slc)
        except TypeError:
            # Likely a slice object.
            try:
                length = slc.stop - slc.start
            except TypeError:
                # Likely a NoneType slice.
                if slc.start is None:
                    if slc.stop > 0:
                        length = length_self
                    elif slc.stop is None:
                        length = length_self
                    else:
                        length = length_self + slc.stop
                elif slc.stop is None:
                    if slc.start > 0:
                        length = length_self - slc.start
                    else:
                        length = abs(slc.start)
                else:
                    raise
            else:
                # If the slice length is greater than the current length, keep the length the same.
                if length > length_self:
                    length = length_self
        else:
            try:
                # Check for boolean slices.
                if slc.dtype == bool:
                    length = slc.sum()
            except AttributeError:
                # Likely a list/tuple.
                pass

        if length < 0:
            # This is using negative indexing. Subtract from the current length.
            length += length_self

        # Source index can be None if the dimension has zero length.
        if ret._src_idx is not None:
            src_idx = ret._src_idx.__getitem__(slc)
        else:
            src_idx = None

        ret.set_size(length, src_idx=src_idx)


def create_src_idx(start, stop, dtype=np.int32):
    return np.arange(start, stop, dtype=dtype)
