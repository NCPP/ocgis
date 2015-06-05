import abc
from collections import OrderedDict
from copy import copy, deepcopy
from operator import mul

import numpy as np

from ocgis import constants
from ocgis.constants import NAME_BOUNDS_DIMENSION_LOWER, NAME_BOUNDS_DIMENSION_UPPER, OCGIS_BOUNDS
from ocgis.util.helpers import get_none_or_1d, get_none_or_2d, get_none_or_slice, \
    get_formatted_slice, get_bounds_from_1d
from ocgis.exc import EmptySubsetError, ResolutionError, BoundsAlreadyAvailableError
from ocgis.interface.base.variable import AbstractValueVariable, AbstractSourcedVariable


class AbstractDimension(object):
    """
    :param dict meta:
    :param str name:
    :param array-like properties:
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def _ndims(self):
        """int"""

    @abc.abstractproperty
    def _attrs_slice(self):
        """sequence of strings"""

    def __init__(self, meta=None, name=None, properties=None):
        self.meta = meta or {}
        self.name = name
        self.properties = properties

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

    def _get_none_or_array_(self, arr, masked=False):
        if self._ndims == 1:
            ret = get_none_or_1d(arr)
        elif self._ndims == 2:
            ret = get_none_or_2d(arr)
        else:
            raise NotImplementedError
        if ret is not None and masked and not isinstance(ret, np.ma.MaskedArray):
            ret = np.ma.array(ret, mask=False)
        return ret

    def _get_sliced_properties_(self, slc):
        if self.properties is not None:
            raise NotImplementedError
        else:
            return None


class AbstractValueDimension(AbstractValueVariable):
    """
    :keyword str name_value: (``=None``) The name of the value for the dimension.
    """
    __metaclass__ = abc.ABCMeta
    _name_value = None

    def __init__(self, *args, **kwargs):
        self.name_value = kwargs.pop('name_value', None)
        AbstractValueVariable.__init__(self, *args, **kwargs)

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


class AbstractUidDimension(AbstractDimension):
    def __init__(self, *args, **kwargs):
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
        self._uid = self._get_none_or_array_(value, masked=True)

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
        kwds_value = ['value', 'name_value', 'units', 'name', 'dtype', 'attrs', 'conform_units_to']
        kwds_uid = ['uid', 'name_uid', 'meta', 'properties', 'name']

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


class VectorDimension(AbstractSourcedVariable, AbstractUidValueDimension):
    _attrs_slice = ('uid', '_value', '_src_idx')
    _ndims = 1

    def __init__(self, *args, **kwargs):
        if kwargs.get('value') is None and kwargs.get('data') is None:
            msg = 'Without a "data" object, "value" is required.'
            raise ValueError(msg)

        self._bounds = None
        self._name_bounds = None
        self._name_bounds_tuple = None

        self.name_bounds_dimension = kwargs.pop('name_bounds_dimension', OCGIS_BOUNDS)
        bounds = kwargs.pop('bounds', None)
        # used for creating name_bounds as well as the name of the bounds dimension in netCDF
        self.name_bounds = kwargs.pop('name_bounds', None)
        self.name_bounds_tuple = kwargs.pop('name_bounds_tuple', None)
        self.axis = kwargs.pop('axis', None)
        # if True, bounds were interpolated. if False, they were loaded from source data. used in conforming units.
        self._has_interpolated_bounds = False

        AbstractSourcedVariable.__init__(self, kwargs.pop('data', None), kwargs.pop('src_idx', None))
        AbstractUidValueDimension.__init__(self, *args, **kwargs)

        # setting bounds requires checking the data type of value set in a superclass.
        self.bounds = bounds

        # conform any units if they provided. check they are not equivalent first
        if self.conform_units_to is not None:
            if not self.conform_units_to.equals(self.cfunits):
                self.cfunits_conform(self.conform_units_to)

    def __len__(self):
        return self.shape[0]

    @property
    def bounds(self):
        # always load the value first. any bounds read from source are set during this process. bounds without values
        # are meaningless!
        self.value

        # if no error is encountered, then the bounds should have been set during loading from source. simply return the
        # value. it will be none, if no bounds were present in the source data.
        return self._bounds

    @bounds.setter
    def bounds(self, value):
        # set the bounds variable.
        self._bounds = get_none_or_2d(value)
        # validate the value
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

    def cfunits_conform(self, to_units):
        """
        Convert and set value and bounds for the dimension object to new units.

        :param to_units: The destination units.
        :type to_units: :class:`cfunits.cfunits.Units`
        """

        # get the original units for bounds conversion. the "cfunits_conform" method updates the object's internal
        # "units" attribute.
        original_units = deepcopy(self.cfunits)
        # call the superclass unit conversion
        AbstractValueVariable.cfunits_conform(self, to_units)
        # if the bounds are already loaded, convert
        if self._bounds is not None:
            AbstractValueVariable.cfunits_conform(self, to_units, value=self._bounds, from_units=original_units)
        # if the bound are not set, they may be interpolated
        elif self.bounds is not None:
            # if the bounds were interpolated, then this should be set to "None" so the units conforming will use the
            # source value units spec.
            if self._has_interpolated_bounds:
                from_units = None
            else:
                from_units = original_units
            # conform the bounds value
            AbstractValueVariable.cfunits_conform(self, to_units, value=self.bounds, from_units=from_units)

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        assert (lower <= upper)

        # # determine if data bounds are contiguous (if bounds exists for the
        # # data). bounds must also have more than one row
        is_contiguous = False
        if self.bounds is not None:
            try:
                if len(set(self.bounds[0, :]).intersection(set(self.bounds[1, :]))) > 0:
                    is_contiguous = True
            except IndexError:
                ## there is likely not a second row
                if self.bounds.shape[0] == 1:
                    pass
                else:
                    raise

        ## subset operation when bounds are not present
        if self.bounds is None or use_bounds == False:
            if closed:
                select = np.logical_and(self.value > lower, self.value < upper)
            else:
                select = np.logical_and(self.value >= lower, self.value <= upper)
        ## subset operation in the presence of bounds
        else:
            ## determine which bound column contains the minimum
            if self.bounds[0, 0] <= self.bounds[0, 1]:
                lower_index = 0
                upper_index = 1
            else:
                lower_index = 1
                upper_index = 0
            ## reference the minimum and maximum bounds
            bounds_min = self.bounds[:, lower_index]
            bounds_max = self.bounds[:, upper_index]

            ## if closed is True, then we are working on a closed interval and
            ## are not concerned if the values at the bounds are equivalent. it
            ## does not matter if the bounds are contiguous.
            if closed:
                select_lower = np.logical_or(bounds_min > lower, bounds_max > lower)
                select_upper = np.logical_or(bounds_min < upper, bounds_max < upper)
            else:
                ## if the bounds are contiguous, then preference is given to the
                ## lower bound to avoid duplicate containers (contiguous bounds
                ## share a coordinate)
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

    def set_extrapolated_bounds(self):
        """Set the bounds variable using extrapolation."""

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        self.bounds = get_bounds_from_1d(self.value)
        self._has_interpolated_bounds = True

    def write_to_netcdf_dataset(self, dataset, unlimited=False, bounds_dimension_name=None, **kwargs):
        """
        Write the dimension and its associated value and bounds to an open netCDF dataset object.

        :param dataset: An open dataset object.
        :type dataset: :class:`netCDF4.Dataset`
        :param bool unlimited: If ``True``, create the dimension on the netCDF object with ``size=None``. See
         http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createDimension.
        :param str bounds_dimension_name: If ``None``, default to
         :attr:`ocgis.interface.base.dimension.base.VectorDimension.name_bounds_dimension`.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` to pass to ``createVariable``. See
         http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
        """

        if self.name is None:
            raise ValueError('Writing to netCDF requires a "name" be set to a string value. It is currently None.')

        bounds_dimension_name = bounds_dimension_name or self.name_bounds_dimension

        if unlimited:
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
                # bounds dimension likely created previously. check for it, then move on
                if bounds_dimension_name not in dataset.dimensions:
                    raise
            kwargs['dimensions'] = (self.name, bounds_dimension_name)
            bounds_variable = dataset.createVariable(self.name_bounds, self.bounds.dtype, **kwargs)
            bounds_variable[:] = self.bounds
            variable.setncattr('bounds', self.name_bounds)

        # data mode issues require that this be last...?
        self.write_attributes_to_netcdf_object(variable)

    def _format_private_value_(self, value):
        value = self._get_none_or_array_(value, masked=False)
        return value

    def _format_slice_state_(self, state, slc):
        state.bounds = get_none_or_slice(state._bounds, (slc, slice(None)))
        return (state)

    def _format_src_idx_(self, value):
        return (self._get_none_or_array_(value))

    def _get_iter_value_bounds_(self):
        return (self.value, self.bounds)

    def _get_uid_(self):
        if self._value is not None:
            shp = self._value.shape[0]
        else:
            shp = self._src_idx.shape[0]
        ret = np.arange(1, shp + 1, dtype=np.int32)
        ret = np.atleast_1d(ret)
        return ret

    def _set_value_from_source_(self):
        if self._value is None:
            raise NotImplementedError
        else:
            self._value = self._value

    def _validate_bounds_(self):
        # # bounds must be two-dimensional
        if self._bounds.shape[1] != 2:
            raise (ValueError('Bounds array must be two-dimensional.'))
        # # bounds and value arrays must have matching data types. if they do
        ## not match, attempt to cast the bounds.
        try:
            assert (self._bounds.dtype == self._value.dtype)
        except AssertionError:
            try:
                self._bounds = np.array(self._bounds, dtype=self._value.dtype)
            except:
                raise (ValueError('Value and bounds data types do not match and types could not be casted.'))
