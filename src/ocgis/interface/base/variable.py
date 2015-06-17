import abc
from collections import OrderedDict
from copy import copy, deepcopy

import numpy as np

from ocgis.api.collection import AbstractCollection
from ocgis.constants import NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE
from ocgis.interface.base.attributes import Attributes
from ocgis.util.helpers import get_iter, iter_array
from ocgis.exc import NoUnitsError, VariableInCollectionError


class AbstractValueVariable(Attributes):
    """
    :param array-like value:
    :param units:
    :type units: str or :class:`cfunits.Units`
    :param :class:`numpy.dtype` dtype:
    :param fill_value:
    :type fill_value: int or float matching type of ``dtype``
    :param str name:
    :param conform_units_to:
    :type units: str or :class:`cfunits.Units`
    :param str alias: An alternate name for the variable used to maintain uniqueness.
    :param dict attrs: A dictionary of arbitrary key-value attributes.
    """

    __metaclass__ = abc.ABCMeta
    _value = None
    _conform_units_to = None

    def __init__(self, value=None, units=None, dtype=None, name=None, conform_units_to=None,
                 alias=None, attrs=None):
        self.name = name
        self.alias = alias or self.name

        Attributes.__init__(self, attrs=attrs)

        # if the units value is not None, then convert to string. cfunits.Units may be easily handled this way without
        # checking for the module presence.
        self.units = str(units) if units is not None else None
        self.conform_units_to = conform_units_to
        self.value = value

        # Default to the value data types and fill values ignoring the provided values.
        if value is None:
            self._dtype = dtype
        else:
            self._dtype = None

    @property
    def cfunits(self):
        # the cfunits-python module is not a dependency of ocgis and should be imported on demand
        from cfunits import Units

        return Units(self.units)

    def _conform_units_to_getter_(self):
        return self._conform_units_to

    def _conform_units_to_setter_(self, value):
        if value is not None:
            from cfunits import Units

            if not isinstance(value, Units):
                value = Units(value)
        self._conform_units_to = value

    conform_units_to = property(_conform_units_to_getter_, _conform_units_to_setter_)

    @property
    def dtype(self):
        if self._dtype is None:
            if self._value is None:
                raise ValueError('dtype not specified at object initialization and value has not been loaded.')
            else:
                ret = self.value.dtype
        else:
            ret = self._dtype
        return ret

    @property
    def shape(self):
        return self.value.shape

    @property
    def value(self):
        if self._value is None:
            self._value = self._format_private_value_(self._get_value_())
        return self._value

    @value.setter
    def value(self, value):
        self._value = self._format_private_value_(value)

    def cfunits_conform(self, to_units, value=None, from_units=None):
        """
        Conform units of value variable in-place using :mod:`cfunits`. If there are an scale or offset parameters in the
        attribute dictionary, they will be removed.

        :param to_units: Target conform units.
        :type t_units: str or :class:`cfunits.Units`
        :param value: Optional value array to use in place of the object's value.
        :type value: :class:`numpy.ma.array`
        :param from_units: Source units to use in place of the object's value.
        :type from_units: str or :class:`cfunits.Units`
        :rtype: np.ndarray
        :raises: NoUnitsError
        """

        from cfunits import Units

        # units are required for conversion
        if self.cfunits == Units(None):
            raise NoUnitsError(self.alias)
        # allow string unit representations to be passed
        if not isinstance(to_units, Units):
            to_units = Units(to_units)
        # pick the value to convert. this is added to keep the import of the units library in the
        # AbstractValueVariable.cfunits property
        convert_value = self._get_to_conform_value_() if value is None else value
        # use the overloaded "from_units" if passed, otherwise use the object-level attribute
        from_units = self.cfunits if from_units is None else from_units
        # units are always converted in place. users need to execute their own deep copies
        self.cfunits.conform(convert_value, from_units, to_units, inplace=True)
        # update the units attribute with the destination units
        if hasattr(to_units, 'calendar'):
            # The string representation of units contains the calendar in the case of time. It only prints the calendar
            # if the value is not None.
            if to_units.calendar is not None:
                str_to_units = Units(to_units.units)
            else:
                str_to_units = to_units
        else:
            str_to_units = to_units
        self.units = str(str_to_units)
        # let the data type load natively from the value array
        self._dtype = None
        # remove any compression attributes if present
        for remove in NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE:
            self.attrs.pop(remove, None)

        return convert_value

    def _format_private_value_(self, value):
        if value is not None:
            # conform the units if a value is passed and the units are not equivalent
            if self.conform_units_to is not None:
                if not self.conform_units_to.equals(self.cfunits):
                    value = self.cfunits_conform(to_units=self.conform_units_to, value=value, from_units=self.cfunits)
        return value

    def _get_to_conform_value_(self):
        """Intended for subclasses to be able to provide a different value array for unit conforming."""
        return self.value

    @abc.abstractmethod
    def _get_value_(self):
        """Return the value field."""


class AbstractSourcedVariable(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, src_idx):
        self._data = data
        self._src_idx = src_idx

    @property
    def _src_idx(self):
        return self.__src_idx

    @_src_idx.setter
    def _src_idx(self, value):
        self.__src_idx = self._format_src_idx_(value)

    def _format_src_idx_(self, value):
        return np.array(value)

    def _get_value_(self):
        if self._value is None:
            self._set_value_from_source_()
        return self._value

    @abc.abstractmethod
    def _set_value_from_source_(self):
        """Should set ``_value`` using the data source and index."""


class Variable(AbstractSourcedVariable, AbstractValueVariable):
    """
    :param name: Representative name for the variable.
    :type name: str
    :param alias: Optional unique name for the variable.
    :type alias: str
    :param units: Variable units. If :mod:`cfunits-python` is installed, this will be
     transformed into a :class:`cfunits.Units` object.
    :type units: str
    :param meta: Optional metadata dictionary or object.
    :type meta: dict or object
    :param uid: Optional unique identifier for the Variable.
    :type uid: int
    :param value: Value associated with the variable.
    :type value: np.ndarray
    :param data: Optional data source if no value is passed.
    :type data: object
    :param did: Optional unique identifier for the data source.
    :type did: int
    :param dtype: Optional data type of the object.
    :type dtype: type
    :param fill_value: Optional fill value for masked array elements.
    :type fill_value: int or float
    :param conform_units_to: Target units for conversion.
    :type conform_units_to: str convertible to :class:`cfunits.Units`
    :param dict attrs: A dictionary of arbitrary key-value attributes.
    """

    def __init__(self, name=None, alias=None, units=None, meta=None, uid=None, value=None, did=None, data=None,
                 conform_units_to=None, dtype=None, fill_value=None, attrs=None):
        self.meta = meta or {}
        self.uid = uid
        self.did = did

        if value is None:
            self._fill_value = fill_value
        else:
            self._fill_value = None

        AbstractSourcedVariable.__init__(self, data, None)
        AbstractValueVariable.__init__(self, value=value, units=units, dtype=dtype, name=name,
                                       conform_units_to=conform_units_to, alias=alias, attrs=attrs)

    def __getitem__(self, slc):
        ret = copy(self)
        if ret._value is not None:
            # store the previous number of dimension to ensure this does not change following a slice
            prev_ndim = ret._value.ndim
            ret._value = self._value[slc]
            if prev_ndim != ret._value.ndim:
                # if the number of dimensions has changed but they are all singleton, add one back in.
                if all([xx == 1 for xx in ret._value.shape]):
                    ret._value = ret._value.reshape(*[1] * prev_ndim)
                else:
                    msg = 'Array has changed shaped following slicing.'
                    raise IndexError(msg)
        return ret

    def __str__(self):
        units = '{0}' if self.units is None else '"{0}"'
        units = units.format(self.units)
        ret = '{0}(name="{1}", alias="{2}", units={3})'.format(self.__class__.__name__, self.alias, self.name, units)
        return ret

    @property
    def fill_value(self):
        if self._fill_value is None:
            if self._value is None:
                raise ValueError('"fill_value" not specified at object initialization and value has not been loaded.')
            else:
                ret = self.value.fill_value
        else:
            ret = self._fill_value
        return ret

    def get_empty_like(self, shape=None):
        """
        Create a variable with an empty value array. The ``data`` object is not copied. Otherwise, all attributes are
        copied. This is useful for cases when the variable needs to be reshaped with all attributes maintained.

        :param shape: If provided, allocate a masked array with the given shape.
        :type shape: tuple of five ints

        >>> shape = (2, 10, 3, 5, 6)

        :rtype: :class:`ocgis.interface.base.variable.Variable`
        """

        if shape is None:
            mask = self.value.mask
        else:
            mask = False
        shape = shape or self.value.shape
        value = np.ma.array(np.zeros(shape), dtype=self.dtype, fill_value=self.fill_value, mask=mask)
        ret = Variable(name=self.name, units=self.units, meta=deepcopy(self.meta), value=value, did=self.did,
                       alias=self.alias, uid=self.uid, attrs=deepcopy(self.attrs))
        return ret

    def iter_melted(self, use_mask=True):
        """
        :param bool use_mask: If ``True`` (the default), do not yield masked values. If ``False``, yield the underlying
         masked data value.
        :returns: A dictionary containing variable values for each index location in the array.
        :rtype: dict
        """

        units = self.units
        uid = self.uid
        did = self.did
        alias = self.alias
        name = self.name
        attrs = self.attrs

        for idx, value in iter_array(self.value, use_mask=use_mask, return_value=True):
            yld = {'value': value, 'units': units, 'uid': uid, 'did': did, 'alias': alias, 'name': name, 'slice': idx,
                   'attrs': attrs}
            yield yld

    def _format_private_value_(self, value):
        # the superclass method does nice things like conform units if appropriate
        value = AbstractValueVariable._format_private_value_(self, value)
        if value is None:
            ret = None
        else:
            if not isinstance(value, np.ma.MaskedArray):
                ret = np.ma.array(value, mask=False)
            else:
                ret = value
        return ret

    def _get_value_(self):
        if self._value is None:
            self._set_value_from_source_()
        return self._value

    def _set_value_from_source_(self):
        # load the value from source using the referenced field
        self._value = self._field._get_value_from_source_(self._data, self.name)
        # ensure the new value has the geometry masked applied
        self._field._set_new_value_mask_(self._field, self._field.spatial.get_mask())


class VariableCollection(AbstractCollection):
    def __init__(self, variables=None):
        super(VariableCollection, self).__init__()

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def add_variable(self, variable, assign_new_uid=False):
        """
        :param :class:`ocgis.interface.base.variable.Variable` :
        :param bool assign_new_uid: If ``True``, assign a new unique identifier to the incoming variable. This will
         modify the variable inplace.
        """
        assert (isinstance(variable, Variable))
        try:
            assert (variable.alias not in self)
        except AssertionError:
            raise VariableInCollectionError(variable)

        if assign_new_uid:
            variable.uid = None

        if variable.uid is None:
            variable.uid = self._storage_id_next
        else:
            assert (variable.uid not in self._storage_id)
        self._storage_id.append(variable.uid)
        self.update({variable.alias: variable})

    def get_sliced_variables(self, slc):
        variables = [v.__getitem__(slc) for v in self.itervalues()]
        ret = VariableCollection(variables=variables)
        return ret

    def iter_columns(self):
        """
        :returns: An iterator over each variable index.
        :rtype: :class:`collections.OrderedDict`
        """

        self_itervalues = self.itervalues
        dmap = {v.alias: v.value.data for v in self_itervalues()}
        for idx in iter_array(self.first().value.data):
            yld = OrderedDict()
            for v in self_itervalues():
                alias = v.alias
                yld[alias] = dmap[alias][idx]
            yld = (idx, yld)
            yield yld

    def iter_melted(self, **kwargs):
        """
        :returns: Call :meth:`~ocgis.Variable.iter_melted` passing ``kwargs`` for each variable in the collection.
        :rtype: see :meth:`~ocgis.Variable.iter_melted`
        """

        for variable in self.itervalues():
            for row in variable.iter_melted(**kwargs):
                yield row


class DerivedVariable(Variable):
    """
    Variable class for derived variables.

    :param dict fdef: The function definition dictionary.

    >>> fdef = {'name': 'mean', 'func': 'mean'}

    :param parents: The parent variables used to derive the current variable.
    :type parents: :class:`ocgis.interface.base.variable.VariableCollection`
    """

    def __init__(self, **kwargs):
        self.fdef = kwargs.pop('fdef')
        self.parents = kwargs.pop('parents', None)

        super(DerivedVariable, self).__init__(**kwargs)

    def iter_melted(self, **kwargs):
        calc_key = self.fdef['func']
        calc_alias = self.fdef['name']

        if self.parents is not None:
            first = self.parents.first()
            name = first.name
            alias = first.alias
        else:
            name, alias = None, None

        for row in super(DerivedVariable, self).iter_melted(**kwargs):
            row['calc_key'] = calc_key
            row['calc_alias'] = calc_alias
            row['name'] = name
            row['alias'] = alias
            yield row
