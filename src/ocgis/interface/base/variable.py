import abc
from collections import OrderedDict
from copy import copy, deepcopy

import numpy as np

from ocgis.api.collection import AbstractCollection
from ocgis.constants import NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE
from ocgis.exc import NoUnitsError, VariableInCollectionError
from ocgis.interface.base.attributes import Attributes
from ocgis.util.helpers import get_iter, iter_array, set_new_value_mask_for_variable
from ocgis.util.units import get_units_object, get_conformed_units, get_are_units_equal


class AbstractVariable(Attributes):
    __metaclass__ = abc.ABCMeta

    def __init__(self, value=None, units=None, dtype=None, name=None, alias=None, attrs=None, conform_units_to=None):
        if conform_units_to is not None and units is None:
            msg = '"units" are required when "conform_units_to" is not None.'
            raise ValueError(msg)

        self._value = None
        self._units = None
        self._conform_units_to = None

        Attributes.__init__(self, attrs=attrs)

        self.name = name
        self.alias = alias or self.name
        self.units = units
        self.conform_units_to = conform_units_to
        self.value = value

        # Default to the value data type ignoring the provided values.
        if value is None:
            self._dtype = dtype
        else:
            self._dtype = None

    @property
    def cfunits(self):
        return get_units_object(self.units)

    @property
    def conform_units_to(self):
        return self._conform_units_to

    @conform_units_to.setter
    def conform_units_to(self, value):
        if value is not None:
            value = get_units_object(value)
        self._conform_units_to = value

    @property
    def dtype(self):
        if self._dtype is None:
            if self._value is None:
                raise ValueError('"dtype" not specified at object initialization and "value" has not been loaded.')
            else:
                ret = self.value.dtype
        else:
            ret = self._dtype
        return ret

    @property
    def shape(self):
        return self.value.shape

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._set_units_(value)

    @property
    def value(self):
        if self._value is None:
            self._set_value_(self._get_value_())
        return self._value

    @value.setter
    def value(self, value):
        self._set_value_(value)

    def cfunits_conform(self, to_units, from_units=None):
        """
        Conform value units in-place. If there are scale or offset parameters in the attribute dictionary, they will
        be removed.

        :param to_units: Target conform units.
        :type to_units: str or units object
        :param from_units: Overload source units.
        :type from_units: str or units object
        :raises: NoUnitsError
        """
        if from_units is None and self.units is None:
            raise NoUnitsError(self)

        # Use overloaded value for source units.
        from_units = self.cfunits if from_units is None else from_units

        # Get the conform value before swapping the units. Conversion inside time dimensions may be negatively affected
        # otherwise.
        to_conform_value = self._get_to_conform_value_()

        # Update the units attribute with the destination units. Do this before conversion to not enter recursion when
        # setting the new value.
        self.units = to_units

        # Conform the units.
        new_value = get_conformed_units(to_conform_value, from_units, to_units)
        self._set_to_conform_value_(new_value)

        # Let the data type load from the value array.
        self._dtype = None
        # Remove any compression attributes if present.
        for remove in NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE:
            self.attrs.pop(remove, None)

    def _get_to_conform_value_(self):
        return self.value

    def _set_to_conform_value_(self, value):
        self.value = value

    def _set_units_(self, value):
        if value is not None:
            value = str(value)
        self._units = value

    def _get_value_(self):
        raise NotImplementedError

    def _set_value_(self, value):
        self._value = value
        if value is not None and self.conform_units_to is not None:
            are_units_equal = get_are_units_equal((self.cfunits, self.conform_units_to))
            if not are_units_equal:
                self.cfunits_conform(self.conform_units_to)


class AbstractSourcedVariable(AbstractVariable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *args, **kwargs):
        self.__src_idx__ = None

        self._request_dataset = kwargs.pop('request_dataset', None)
        self._src_idx = kwargs.pop('src_idx', None)

        super(AbstractSourcedVariable, self).__init__(*args, **kwargs)

    @property
    def _src_idx(self):
        return self._get_src_idx_()

    @_src_idx.setter
    def _src_idx(self, value):
        self._set_src_idx_(value)

    def _get_src_idx_(self):
        return self.__src_idx__

    def _set_src_idx_(self, value):
        if value is not None:
            value = np.atleast_1d(value)
        self.__src_idx__ = value

    def _get_value_(self):
        return self._get_value_from_source_()

    @abc.abstractmethod
    def _get_value_from_source_(self):
        """Get value data from a request dataset."""


class Variable(AbstractSourcedVariable):
    """
    :param name: Representative name for the variable.
    :type name: str
    :param alias: Optional unique name for the variable.
    :type alias: str
    :param units: Optional units for the variable.
    :type units: str
    :param meta: Optional dictionary of arbitrary metadata.
    :type meta: dict or object
    :param uid: Optional unique identifier for the Variable.
    :type uid: int
    :param value: Value associated with the variable. Arrays will be always be transformed to masked arrays.
    :type value: :class:`numpy.ndarray`
    :param request_dataset: Optional data source if no value is passed.
    :type request_dataset: :class:`~ocgis.RequestDataset`
    :param dtype: Optional data type of the object.
    :type dtype: type
    :param fill_value: Optional fill value for masked array elements.
    :type fill_value: int or float
    :param dict attrs: A dictionary of arbitrary key-value attributes.
    :param conform_units_to: Destination units for the value data.
    :type conform_units_to: str or units object
    """

    def __init__(self, name=None, alias=None, units=None, meta=None, uid=None, value=None, request_dataset=None,
                 dtype=None, fill_value=None, attrs=None, conform_units_to=None):
        self._field = None

        self.meta = meta or {}
        self.uid = uid

        # Use the fill value associated with value array.
        if value is None:
            self._fill_value = fill_value
        else:
            self._fill_value = None

        # "src_idx" is None because source indices are pulled from the associated field object when loading data.
        AbstractSourcedVariable.__init__(self, value=value, units=units, dtype=dtype, name=name, alias=alias,
                                         attrs=attrs, request_dataset=request_dataset, src_idx=None,
                                         conform_units_to=conform_units_to)

    def __getitem__(self, slc):
        ret = copy(self)
        if ret._value is not None:
            # Store the previous number of dimensions to ensure this does not change following a slice.
            prev_ndim = ret._value.ndim
            ret._value = self._value[slc]
            if prev_ndim != ret._value.ndim:
                # If the number of dimensions has changed but they are all singleton, add one back in.
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
        ret = Variable(name=self.name, units=self.units, meta=deepcopy(self.meta), value=value, alias=self.alias,
                       uid=self.uid, attrs=deepcopy(self.attrs))
        return ret

    def iter_melted(self, use_mask=True):
        """
        :param bool use_mask: If ``True``, do not yield masked values. If ``False``, yield the underlying masked data
         value.
        :returns: A dictionary containing variable values for the flattened value array.
        :rtype: dict
        """

        units = self.units
        uid = self.uid
        alias = self.alias
        name = self.name
        attrs = self.attrs
        meta = self.meta

        for _, value in iter_array(self.value, use_mask=use_mask, return_value=True):
            yld = {'value': value, 'units': units, 'uid': uid, 'alias': alias, 'name': name, 'attrs': attrs,
                   'meta': meta}
            yield yld

    def _set_value_(self, value):
        if value is None:
            res = None
        else:
            if not isinstance(value, np.ma.MaskedArray):
                res = np.ma.array(value, mask=False)
            else:
                # Indexing into masks should always be supported. If the mask is a scalar, convert to a boolean array.
                if np.isscalar(value.mask):
                    new_mask = np.empty_like(value, dtype=bool)
                    new_mask[:] = value.mask
                    res = np.ma.array(value.data, mask=new_mask)
                else:
                    res = value
        super(Variable, self)._set_value_(res)

    def _get_value_from_source_(self):
        # Load the value from source using the referenced field.
        value = self._field._get_value_from_source_(self._request_dataset, self.name)
        # Apply the geometry mask.
        set_new_value_mask_for_variable(self._field.spatial.get_mask(), self._field.shape, value)
        return value


class VariableCollection(AbstractCollection):
    def __init__(self, variables=None):
        super(VariableCollection, self).__init__()

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def add_variable(self, variable, assign_new_uid=False):
        """
        :param variable: The variable to add to the collection.
        :type: :class:`ocgis.interface.base.variable.Variable`
        :param bool assign_new_uid: If ``True``, assign a new unique identifier to the incoming variable. This will
         modify the variable in-place.
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
        :param kwargs: Dictionary of keyword arguments to pass to each variable's ``iter_melted`` method.
        :type kwargs: dict
        :returns: Call :meth:`~ocgis.Variable.iter_melted` passing ``kwargs`` for each variable in the collection.
        :rtype: See :meth:`~ocgis.Variable.iter_melted`
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
