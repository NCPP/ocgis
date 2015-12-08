import abc
from collections import OrderedDict
from copy import copy
from operator import mul

import numpy as np

from ocgis import constants
from ocgis.constants import NAME_BOUNDS_DIMENSION_LOWER, NAME_BOUNDS_DIMENSION_UPPER, OCGIS_BOUNDS
from ocgis.exc import EmptySubsetError, ResolutionError, BoundsAlreadyAvailableError
from ocgis.interface.base.variable import AbstractSourcedVariable
from ocgis.util.helpers import get_none_or_1d, get_none_or_2d, get_none_or_slice, \
    get_formatted_slice, get_bounds_from_1d
from ocgis.util.units import get_conformed_units, get_are_units_equal


class AbstractDimension(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def _ndims(self):
        """int"""

    @abc.abstractproperty
    def _attrs_slice(self):
        """sequence of strings"""

    def __init__(self, meta=None, name=None, properties=None, unlimited=False):
        self.meta = meta or {}
        self.name = name
        self.properties = properties
        self.unlimited = unlimited

        if self.properties is not None:
            assert isinstance(self.properties, np.ndarray)

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, self._ndims)
        ret = copy(self)
        for attr in self._attrs_slice:
            ref_set = get_none_or_slice(getattr(ret, attr), slc)
            setattr(ret, attr, ref_set)
        ret.properties = self._get_sliced_properties_(slc)
        ret = self._format_slice_state_(ret, slc)
        return ret

    def get_iter(self):
        raise NotImplementedError

    def _format_slice_state_(self, state, slc):
        return state

    def _get_sliced_properties_(self, slc):
        if self.properties is not None:
            raise NotImplementedError
        else:
            return None


class AbstractValueDimension(AbstractSourcedVariable):
    """
    :keyword str name_value: (``=None``) The name of the value for the dimension.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        self._name_value = None

        self.name_value = kwargs.pop('name_value', None)

        AbstractSourcedVariable.__init__(self, *args, **kwargs)

    @property
    def name_value(self):
        if self._name_value is None:
            ret = self.name
        else:
            ret = self._name_value
        return ret

    @name_value.setter
    def name_value(self, value):
        self._name_value = value

    def _get_value_from_source_(self):
        if self._value is None:
            raise NotImplementedError
        return self._value


class AbstractUidDimension(AbstractDimension):
    def __init__(self, *args, **kwargs):
        self._name_uid = None
        self._uid = None

        self.uid = kwargs.pop('uid', None)
        self.name_uid = kwargs.pop('name_uid', None)

        super(AbstractUidDimension, self).__init__(*args, **kwargs)

    @property
    def name_uid(self):
        if self._name_uid is None:
            ret = '{0}_uid'.format(self.name)
        else:
            ret = self._name_uid
        return ret

    @name_uid.setter
    def name_uid(self, value):
        self._name_uid = value

    @property
    def uid(self):
        if self._uid is None:
            self._uid = self._get_uid_()
        return self._uid

    @uid.setter
    def uid(self, value):
        self._uid = get_none_or_array(value, self._ndims, masked=True)

    def _get_uid_(self):
        if self.value is None:
            ret = None
        else:
            n = reduce(mul, self.value.shape)
            # The unique identifier is set to 32-bit to decrease memory.
            ret = np.arange(1, n + 1, dtype=np.int32).reshape(self.value.shape)
            ret = np.ma.array(ret, mask=False)
        return ret


class AbstractUidValueDimension(AbstractValueDimension, AbstractUidDimension):
    def __init__(self, *args, **kwargs):
        kwds_value = ['value', 'name_value', 'units', 'name', 'dtype', 'attrs', 'src_idx', 'request_dataset',
                      'conform_units_to']
        kwds_uid = ['uid', 'name_uid', 'meta', 'properties', 'name', 'unlimited']

        kwds_all = kwds_value + kwds_uid
        for key in kwargs.keys():
            try:
                assert key in kwds_all
            except AssertionError:
                msg = '"{0}" is not a valid keyword argument for "{1}".'
                raise ValueError(msg.format(key, self.__class__.__name__))

        kwds_value = {key: kwargs.get(key, None) for key in kwds_value}
        kwds_uid = {key: kwargs.get(key, None) for key in kwds_uid}

        AbstractValueDimension.__init__(self, *args, **kwds_value)
        AbstractUidDimension.__init__(self, *args, **kwds_uid)


class VectorDimension(AbstractUidValueDimension):
    """
    :keyword str alias: (``=None``) An alternate name value. If ``None``, defaults to ``name``.
    :keyword dict attrs: (``=None``) A dictionary of arbitrary key-value attributes.
    :keyword str axis: (``=None``) Name of the axis. Possible values are ``'R'``, ``'T'``, ``'X'``,``'Y'``, and ``'Z'``.
    :keyword conform_units_to: (``=None``) If provided, conform the value data to match these units.
    :type conform_units_to: same as ``units``
    :keyword dtype: (``=None``) Data type for the dimension. If ``None``, defaults to ``value``'s data type.
    :type dtype: :class:`numpy.dtype`
    :keyword dict meta: (``=None``) Dictionary of arbitrary metadata.
    :keyword str name: (``=None``) Name of the dimension.
    :keyword str name_bounds: (``=None``) Name of the bounds data.
    :keyword str name_bounds_dimension: (``=None``) Name of the bounds dimension.
    :keyword str name_bounds_tuple: (``=None``) Tuple of strings for constructing bounds name headers.
    :keyword str name_uid: (``=None``) Name of the unique identifier.
    :keyword str name_value: (``=None``) Name of the value data.
    :keyword properties: (``=None``) Structure array of property values.
    :type properties: :class:`numpy.ndarray`
    :keyword request_dataset: (``=None``) Source request dataset to use for value loading.
    :type request_dataset: :class:`~ocgis.RequestDataset`
    :keyword src_idx: (``=None``) Data to use for loading from source data.
    :type src_idx: Typically a :class:`numpy.ndarray`.
    :keyword uid: (``=None``) Integer unique identifiers for the elements in the dimension.
    :type uid: :class:`numpy.ndarray`
    :keyword units: (``=None``) Units for the dimension.
    :type units: str or units object
    :keyword unlimited: (``=None``) If ``True``, the dimension is unlimited and may be expanded.
    :type unlimited: bool
    :keyword value: (``=None``) Value associated with the dimension (i.e. time float values).
    :type value: :class:`numpy.ndarray`
    """
    _attrs_slice = ('uid', '_value', '_src_idx')
    _ndims = 1

    def __init__(self, *args, **kwargs):
        if kwargs.get('value') is None and kwargs.get('request_dataset') is None:
            msg = 'Without a "request_dataset" object, "value" is required.'
            raise ValueError(msg)

        self._bounds = None
        self._name_bounds = None
        self._name_bounds_tuple = None
        self._original_units = None
        # If True, bounds should always be None.
        self._has_removed_bounds = False

        self.name_bounds_dimension = kwargs.pop('name_bounds_dimension', OCGIS_BOUNDS)
        bounds = kwargs.pop('bounds', None)
        # Used for creating name_bounds as well as the name of the bounds dimension in netCDF.
        self.name_bounds = kwargs.pop('name_bounds', None)
        self.name_bounds_tuple = kwargs.pop('name_bounds_tuple', None)
        self.axis = kwargs.pop('axis', None)

        AbstractUidValueDimension.__init__(self, *args, **kwargs)

        # Setting bounds requires checking the data type of value set in a superclass.
        self.bounds = bounds

    def __len__(self):
        return self.shape[0]

    @property
    def bounds(self):
        if self._bounds is None and not self._has_removed_bounds:
            # Always load the value first.
            assert self.value is not None
            self.bounds = self._get_bounds_from_source_()
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        self._bounds = get_none_or_2d(value)
        if self._bounds is not None and self._original_units is not None:
            are_units_equal = get_are_units_equal((self.units, self._original_units))
            if not are_units_equal:
                self._bounds = get_conformed_units(self._bounds, self._original_units, self.conform_units_to)
        if value is not None:
            self._validate_bounds_()

    @property
    def extent(self):
        if self.bounds is None:
            target = self.value
        else:
            target = self.bounds
        return target.min(), target.max()

    @property
    def name_bounds(self):
        if self._name_bounds is None:
            ret = '{0}_{1}'.format(self.name, self.name_bounds_dimension)
        else:
            ret = self._name_bounds
        return ret

    @name_bounds.setter
    def name_bounds(self, value):
        self._name_bounds = value

    @property
    def name_bounds_tuple(self):
        if self._name_bounds_tuple is None:
            ret = tuple(['{0}_{1}'.format(prefix, self.name) for prefix in [NAME_BOUNDS_DIMENSION_LOWER,
                                                                            NAME_BOUNDS_DIMENSION_UPPER]])
        else:
            ret = self._name_bounds_tuple
        return ret

    @name_bounds_tuple.setter
    def name_bounds_tuple(self, value):
        if value is not None:
            value = tuple(value)
            assert len(value) == 2
        self._name_bounds_tuple = value

    @property
    def resolution(self):
        if self.bounds is None and self.value.shape[0] < 2:
            msg = 'With no bounds and a single coordinate, approximate resolution may not be determined.'
            raise ResolutionError(msg)
        elif self.bounds is None:
            res_array = np.diff(self.value[0:constants.RESOLUTION_LIMIT])
        else:
            res_bounds = self.bounds[0:constants.RESOLUTION_LIMIT]
            res_array = res_bounds[:, 1] - res_bounds[:, 0]
        ret = np.abs(res_array).mean()
        return ret

    @property
    def shape(self):
        return self.uid.shape

    def cfunits_conform(self, *args, **kwargs):
        # Get the from units before conforming the value. The units are changed in the value conform.
        from_units = kwargs.get('from_units') or self.cfunits
        # Store the original units to use for bounds conversion.
        self._original_units = self.cfunits
        # Conform the value.
        AbstractSourcedVariable.cfunits_conform(self, *args, **kwargs)

        # Conform the units
        if self._bounds is not None:
            self._bounds = get_conformed_units(self._bounds, from_units, args[0])

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        assert (lower <= upper)

        # Determine if data bounds are contiguous (if bounds exists for the data). Bounds must also have more than one
        # row.
        is_contiguous = False
        if self.bounds is not None:
            try:
                if len(set(self.bounds[0, :]).intersection(set(self.bounds[1, :]))) > 0:
                    is_contiguous = True
            except IndexError:
                # There is likely not a second row.
                if self.bounds.shape[0] == 1:
                    pass
                else:
                    raise

        # Subset operation when bounds are not present.
        if self.bounds is None or use_bounds == False:
            if closed:
                select = np.logical_and(self.value > lower, self.value < upper)
            else:
                select = np.logical_and(self.value >= lower, self.value <= upper)
        # Subset operation in the presence of bounds.
        else:
            # Determine which bound column contains the minimum.
            if self.bounds[0, 0] <= self.bounds[0, 1]:
                lower_index = 0
                upper_index = 1
            else:
                lower_index = 1
                upper_index = 0
            # Reference the minimum and maximum bounds.
            bounds_min = self.bounds[:, lower_index]
            bounds_max = self.bounds[:, upper_index]

            # If closed is True, then we are working on a closed interval and are not concerned if the values at the
            # bounds are equivalent. It does not matter if the bounds are contiguous.
            if closed:
                select_lower = np.logical_or(bounds_min > lower, bounds_max > lower)
                select_upper = np.logical_or(bounds_min < upper, bounds_max < upper)
            else:
                # If the bounds are contiguous, then preference is given to the lower bound to avoid duplicate
                # containers (contiguous bounds share a coordinate)
                if is_contiguous:
                    select_lower = np.logical_or(bounds_min >= lower, bounds_max > lower)
                    select_upper = np.logical_or(bounds_min <= upper, bounds_max < upper)
                else:
                    select_lower = np.logical_or(bounds_min >= lower, bounds_max >= lower)
                    select_upper = np.logical_or(bounds_min <= upper, bounds_max <= upper)
            select = np.logical_and(select_lower, select_upper)

        if select.any() == False:
            raise (EmptySubsetError(origin=self.name))

        ret = self[select]

        if return_indices:
            indices = np.arange(select.shape[0])
            ret = (ret, indices[select])

        return ret

    def get_iter(self, with_bounds=True):
        ref_value, ref_bounds = self._get_iter_value_bounds_()

        if ref_bounds is None:
            has_bounds = False
        else:
            has_bounds = True

        ref_uid = self.uid
        ref_name_value = self.name_value

        if self.name_value is None:
            msg = 'The "name_value" attribute is required for iteration.'
            raise ValueError(msg)

        ref_name_uid = self.name_uid
        ref_name_bounds_lower, ref_name_bounds_upper = self.name_bounds_tuple

        for ii in range(self.value.shape[0]):
            yld = OrderedDict([(ref_name_uid, ref_uid[ii]), (ref_name_value, ref_value[ii])])
            if with_bounds:
                if has_bounds:
                    ref_name_bounds_lower_value = ref_bounds[ii, 0]
                    ref_name_bounds_upper_value = ref_bounds[ii, 1]
                else:
                    ref_name_bounds_lower_value = None
                    ref_name_bounds_upper_value = None
                yld[ref_name_bounds_lower] = ref_name_bounds_lower_value
                yld[ref_name_bounds_upper] = ref_name_bounds_upper_value
            yield ii, yld

    def get_report(self):
        """
        :returns: A sequence of strings suitable for printing.
        :rtype: list[str, ...]
        """
        lines = ['Name = {0}'.format(self.name),
                 'Count = {0}'.format(self.value.shape[0])]
        if self.bounds is None:
            has_bounds = False
        else:
            has_bounds = True
        lines.append('Has Bounds = {0}'.format(has_bounds))
        lines.append('Data Type = {0}'.format(self.dtype))
        return lines

    def remove_bounds(self):
        self.bounds = None
        self._has_removed_bounds = True

    def set_extrapolated_bounds(self):
        """Set the bounds variable using extrapolation."""

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        self.bounds = get_bounds_from_1d(self.value)

    def write_netcdf(self, dataset, bounds_dimension_name=None, **kwargs):
        """
        Write the dimension and its associated value and bounds to an open netCDF dataset object.

        :param dataset: An open dataset object.
        :type dataset: :class:`netCDF4.Dataset`
        :param str bounds_dimension_name: If ``None``, default to
         :attr:`ocgis.interface.base.dimension.base.VectorDimension.name_bounds_dimension`.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` to pass to ``createVariable``. See
         http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        """

        if self.name is None:
            raise ValueError('Writing to netCDF requires a "name" be set to a string value. It is currently None.')

        bounds_dimension_name = bounds_dimension_name or self.name_bounds_dimension

        if self.unlimited:
            size = None
        else:
            size = self.shape[0]
        dataset.createDimension(self.name, size=size)
        kwargs['dimensions'] = (self.name,)
        variable = dataset.createVariable(self.name_value, self.value.dtype, **kwargs)
        variable[:] = self.value
        variable.axis = self.axis if self.axis is not None else ''

        if self.bounds is not None:
            try:
                dataset.createDimension(bounds_dimension_name, size=2)
            except RuntimeError:
                # Bounds dimension likely created previously. Check for it, then move on.
                if bounds_dimension_name not in dataset.dimensions:
                    raise
            kwargs['dimensions'] = (self.name, bounds_dimension_name)
            bounds_variable = dataset.createVariable(self.name_bounds, self.bounds.dtype, **kwargs)
            bounds_variable[:] = self.bounds
            variable.setncattr('bounds', self.name_bounds)

        # HACK: Data mode issues require that this be last...?
        self.write_attributes_to_netcdf_object(variable)

    def _format_slice_state_(self, state, slc):
        state.bounds = get_none_or_slice(state._bounds, (slc, slice(None)))
        return state

    def _get_iter_value_bounds_(self):
        return self.value, self.bounds

    def _get_uid_(self):
        if self._value is not None:
            shp = self._value.shape[0]
        else:
            shp = self._src_idx.shape[0]
        ret = np.arange(1, shp + 1, dtype=np.int32)
        ret = np.atleast_1d(ret)
        return ret

    def _get_bounds_from_source_(self):
        return self._bounds

    def _validate_bounds_(self):
        # Bounds must be two-dimensional.
        if self._bounds.shape[1] != 2:
            raise ValueError('Bounds array must be two-dimensional.')
        # Bounds and value arrays must have matching data types. If they do not match, attempt to cast the bounds.
        try:
            assert (self._bounds.dtype == self._value.dtype)
        except AssertionError:
            try:
                self._bounds = np.array(self._bounds, dtype=self._value.dtype)
            except:
                raise ValueError('Value and bounds data types do not match and types could not be casted.')

    def _set_value_(self, value):
        value = get_none_or_array(value, self._ndims, masked=False)
        AbstractSourcedVariable._set_value_(self, value)


def get_none_or_array(arr, ndim, masked=False):
    if ndim == 1:
        ret = get_none_or_1d(arr)
    elif ndim == 2:
        ret = get_none_or_2d(arr)
    else:
        raise NotImplementedError
    if ret is not None and masked and not isinstance(ret, np.ma.MaskedArray):
        ret = np.ma.array(ret, mask=False)
    return ret
