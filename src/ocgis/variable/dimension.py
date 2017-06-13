import numpy as np
import six

from ocgis import constants, vm
from ocgis.base import AbstractNamedObject, raise_if_empty
from ocgis.constants import DataType
from ocgis.util.helpers import get_formatted_slice
from ocgis.vmachine.mpi import get_global_to_local_slice, get_nonempty_ranks


class Dimension(AbstractNamedObject):
    """
    A dimension tracks the count of elements along an axis in a multi-dimensional array. All :class:`~ocgis.Variable`
    objects use dimensions. Dimensions are used to track global and local bounds when running and parallel. They also
    track source indices allowed data to be sliced without loading from source. See 
    https://en.wikipedia.org/wiki/Dimension for an overview of dimensions.
    
    :param str name: The dimension's name.
    :param int size: The dimension's size. Set to ``None`` if this dimension is unlimited. 
    :param size_current: The dimension's current size. The current size is needed to track sizes if the dimension is
     unlimited.
    :param src_idx: An one-dimensional, integer array containing the source indices for a "dimensioned" element. If
     ``'auto'``, generate the index array automatically from the dimension size (not applicable for unlimited
     dimensions). 
    :type src_idx: :class:`numpy.ndarray` | ``str``
    :param bool dist: If ``True``, this dimension is distributed. Used by :class:`ocgis.OcgDist` to generate the 
     parallel distribution.
    :param bool is_empty: If ``True``, the dimension is empty on the current rank. 
    :param source_name: See :class:`~ocgis.base.AbstractNamedObject`.
    :param aliases: See :class:`~ocgis.base.AbstractNamedObject`.
    :param uid: See :class:`~ocgis.base.AbstractNamedObject`.
    
    **Example Code:**
    
    >>> # Create standard dimension.
    >>> dim = Dimension('the_dim', size=5)
    >>> assert dim.size == 5 and len(dim) == 5
    
    >>> # Create an unlimited dimension.
    >>> udim = Dimension('unlimited_dim', size_current=10)
    >>> assert udim.size is None and len(udim) == 10 and udim.size_current == 10
    """

    def __init__(self, name, size=None, size_current=None, src_idx=None, dist=False, is_empty=False,
                 source_name=constants.UNINITIALIZED, aliases=None, uid=None):
        if isinstance(src_idx, six.string_types):
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
        """
        :param other: The other dimension.
        :type other: :class:`~ocgis.Dimension`
        :return: ``True`` if dimension objects are equal.
        :rtype: bool
        """

        ret = True
        skip = ('__src_idx__', '_aliases', '_source_name', '_bounds_local', '_bounds_global')
        for k, v in list(self.__dict__.items()):
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
        """
        Dimensions may be sliced like other sliceable Python objects. A shallow copy of the dimension is created before
        slicing. Use :meth:`~ocgis.Dimension.get_distributed_slice` for parallel slicing.
        
        :param slc: A :class:`slice`-like object.
        :rtype: :class:`~ocgis.Dimension`
        :raises: IndexError
         
        >>> dim = Dimension('five', 5)
        >>> sub = dim[2:4]
        >>> assert len(sub) == 2
        >>> assert id(dim) != id(sub)
        """

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
        """
        Get or set the global bounds for the dimension across all non-empty ranks.
        
        ===== ================
        Index Description
        ===== ================
        0     The lower bound.
        1     The upper bound.
        ===== ================
        
        :rtype: ``tuple(int, int)``
        """

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
        """
        Get or set the rank-local bounds for the dimension.

        ===== ================
        Index Description
        ===== ================
        0     The lower bound.
        1     The upper bound.
        ===== ================

        :rtype: ``tuple(int, int)``
        """

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
        """
        :return: ``True`` if the dimension is empty. Allows for the creation of empty objects.
        :rtype: bool
        """
        return self._is_empty

    @property
    def is_unlimited(self):
        """
        :return: ``True`` if the dimension is unlimited.
        :rtype: bool
        """
        if self.size is None:
            ret = True
        else:
            ret = False
        return ret

    @property
    def size(self):
        """
        :return: The dimension's size.
        :rtype: ``int`` | ``None`` if unlimited 
        """
        return self._size

    @property
    def size_current(self):
        """
        :return: The current size of the dimension. Needed to track sizes for unlimited dimensions.
        :rtype: int 
        """
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
        """
        Convert the dimension to an empty dimension.
        
        :raises: ValueError
        """

        if not self.dist:
            raise ValueError('Only distributed dimensions may be converted to empty.')

        self._bounds_local = (0, 0)
        self._bounds_global = (0, 0)
        self._src_idx = None
        self._is_empty = True
        self._size = 0
        self._size_current = 0

    def get_distributed_slice(self, slc):
        """
        Slice the dimension in parallel. The sliced dimension object is a shallow copy. The returned dimension may be
        empty.
        
        :param slc: A :class:`slice`-like object or a fancy slice. If this is a fancy slice, ``slc`` must be
         processor-local. If the fancy slice uses integer indices, the indices must be local. In other words, a fancy
         ``slc`` is not manipulated or redistributed prior to slicing.
        :rtype: :class:`~ocgis.Dimension`
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """

        raise_if_empty(self)

        slc = get_formatted_slice(slc, 1)[0]
        is_fancy = not isinstance(slc, slice)

        if not is_fancy and slc == slice(None):
            ret = self.copy()
        # Use standard slicing for non-distributed dimensions.
        elif not self.dist:
            ret = self[slc]
        else:
            if is_fancy:
                local_slc = slc
            else:
                local_slc = get_global_to_local_slice((slc.start, slc.stop), self.bounds_local)
                if local_slc is not None:
                    local_slc = slice(*local_slc)
            # Slice does not overlap local bounds. The dimension is now empty with size 0.
            if local_slc is None:
                ret = self.copy()
                ret.convert_to_empty()
                dimension_size = 0
            # Slice overlaps so do a slice on the dimension using the local slice.
            else:
                ret = self[local_slc]
                dimension_size = len(ret)
            assert dimension_size >= 0
            dimension_sizes = vm.gather(dimension_size)
            if vm.rank == 0:
                sum_dimension_size = 0
                for ds in dimension_sizes:
                    try:
                        sum_dimension_size += ds
                    except TypeError:
                        pass
                bounds_global = (0, sum_dimension_size)
            else:
                bounds_global = None
            bounds_global = vm.bcast(bounds_global)
            if not ret.is_empty:
                ret.bounds_global = bounds_global

            # Normalize the local bounds on live ranks.
            inner_live_ranks = get_nonempty_ranks(ret, vm)
            with vm.scoped('bounds normalization', inner_live_ranks):
                if not vm.is_null:
                    if vm.rank == 0:
                        adjust = len(ret)
                    else:
                        adjust = None
                    adjust = vm.bcast(adjust)
                    for current_rank in vm.ranks:
                        if vm.rank == current_rank:
                            if vm.rank != 0:
                                ret.bounds_local = [b + adjust for b in ret.bounds_local]
                                adjust += len(ret)
                        vm.barrier()
                        adjust = vm.bcast(adjust, root=current_rank)
        return ret

    def set_size(self, value, src_idx=None):
        """
        Set the dimension's size.
        
        :param value: The new size of the dimension. Allows setting to ``None`` to convert to an unlimited dimension.
        :type value: ``int`` | ``None``
        :param src_idx: The new source index. If ``'auto'``, create a default source index.
        :raises: ValueError
        """

        if value is not None:
            if isinstance(src_idx, six.string_types) and src_idx == 'auto':
                src_idx = create_src_idx(0, value, dtype=DataType.DIMENSION_SRC_INDEX)
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
                    if slc.stop is not None and slc.stop > 0:
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
