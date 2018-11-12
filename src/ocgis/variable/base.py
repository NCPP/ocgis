import abc
import itertools
from abc import abstractmethod
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
from numpy.core.multiarray import ndarray
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant
from ocgis import constants, vm
from ocgis.base import AbstractNamedObject, get_dimension_names, get_variable_names, get_variables, iter_dict_slices, \
    orphaned, raise_if_empty
from ocgis.collection.base import AbstractCollection
from ocgis.constants import HeaderName, KeywordArgument, DriverKey
from ocgis.driver.dimension_map import has_bounds
from ocgis.environment import get_dtype, env
from ocgis.exc import VariableInCollectionError, BoundsAlreadyAvailableError, EmptySubsetError, \
    ResolutionError, NoUnitsError, DimensionsRequiredError, DimensionMismatchError, MaskedDataFound, \
    PayloadProtectedError
from ocgis.util.helpers import get_iter, get_formatted_slice, get_bounds_from_1d, get_extrapolated_corners_esmf, \
    create_ocgis_corners_from_esmf_corners, is_crs_variable, arange_from_bool_ndarray, dict_first, is_xarray
from ocgis.util.helpers import is_auto_dtype
from ocgis.util.units import get_units_object, get_conformed_units
from ocgis.variable.attributes import Attributes
from ocgis.variable.dimension import Dimension
from ocgis.variable.iterator import Iterator
from ocgis.vmachine.mpi import create_nd_slices, get_global_to_local_slice


@six.add_metaclass(abc.ABCMeta)
class AbstractContainer(AbstractNamedObject):
    """
    Base class for objects with a parent.
    
    .. note:: Accepts all parameters to :class:`~ocgis.base.AbstractNamedObject`.

    Additional keyword arguments are:

    :keyword parent: (``=None``) The parent collection for this container. A variable will always become a member of its
     parent.
    :type parent: ``None`` | :class:`~ocgis.VariableCollection`
    :keyword is_empty: (``=None``) Set to True if this is an empty object.
    :type is_empty: None | bool
    """

    def __init__(self, name, aliases=None, source_name=constants.UNINITIALIZED, parent=None, uid=None, is_empty=None):
        self._is_empty = is_empty
        self._parent = parent
        super(AbstractContainer, self).__init__(name, aliases=aliases, source_name=source_name, uid=uid)

    def __getitem__(self, slc):
        """
        :param slc: Standard slicing syntax or a dictionary slice.
        :return: Slice the object and return a shallow copy. 
        :rtype: :class:`~ocgis.variable.base.AbstractContainer`
        """
        ret, slc = self._getitem_initialize_(slc)
        if self._parent is None:
            self._getitem_main_(ret, slc)
            self._getitem_finalize_(ret, slc)
        else:
            if not isinstance(slc, dict):
                slc = get_dslice(self.dimensions, slc)
            new_parent = ret.parent[slc]
            ret = new_parent[ret.name]
        return ret

    @property
    def dimensions(self):
        """
        :return: A dimension dictionary containing all dimensions on associated with variables in the collection.
        :rtype: :class:`~collections.OrderedDict`
        """
        return self._get_dimensions_()

    @property
    def dims(self):
        #tdk: DOC
        return self.dimensions

    @property
    def driver(self):
        """Get the parent's driver class or object."""
        return self.parent.driver

    @property
    def group(self):
        """
        :return: The group index in the parent/child hierarchy. Returns ``None`` if this collection is the head.
        :rtype: ``None`` | :class:`list` of :class:`str`
        """
        curr = self.parent
        # Collections may not always have a parent, unlike variables.
        if curr is None:
            ret = None
        else:
            ret = [curr.source_name]
            while True:
                if curr.parent is None:
                    break
                else:
                    curr = curr.parent
                    ret.append(curr.source_name)
            ret.reverse()
        return ret

    @property
    def has_initialized_parent(self):
        """
        :return: ``True`` if the object's parent has not been initialized.
        :rtype: bool
        """
        return self._parent is not None

    @property
    def is_empty(self):
        """
        :return: ``True`` if the object is empty..
        :rtype: bool
        """
        return self._get_is_empty_()

    @property
    def parent(self):
        """
        Get or set the parent collection.
        
        :rtype: :class:`ocgis.VariableCollection`
        """
        if self._parent is None:
            self._initialize_parent_()
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @abstractmethod
    def get_mask(self, *args, **kwarga):
        """
        :return: The object's mask as a boolean array with same dimension as the object.
        :rtype: :class:`numpy.ndarray`
        """
        raise NotImplementedError

    @abstractmethod
    def set_mask(self, mask, **kwargs):
        """
        Set the object's mask.
        
        :param mask: A boolean mask array or ``None`` to remove the mask.
        """
        raise NotImplementedError

    def set_name(self, name, aliases=None):
        """
        Set the name for the object.

        See :class:`~ocgis.base.AbstractNamedObject` for documentation.
        """
        parent = self._parent
        if parent is not None and self.name in parent:
            parent[name] = parent.pop(self.name)
        super(AbstractContainer, self).set_name(name, aliases=aliases)

    @abstractmethod
    def _get_dimensions_(self):
        raise NotImplementedError

    @staticmethod
    def _get_parent_class_():
        from ocgis import Field
        return Field

    def _getitem_initialize_(self, slc):
        try:
            slc = get_formatted_slice(slc, self.ndim)
        except (NotImplementedError, IndexError) as e:
            # Assume it is a dictionary slice.
            try:
                slc = {k: get_formatted_slice(v, 1)[0] for k, v in list(slc.items())}
            except:
                raise e
        return self, slc

    @abstractmethod
    def _get_is_empty_(self):
        raise NotImplementedError

    def _getitem_main_(self, ret, slc):
        """Perform major slicing operations in-place."""

    def _getitem_finalize_(self, ret, slc):
        """Finalize the returned sliced object in-place."""

    @abstractmethod
    def _initialize_parent_(self, *args, **kwargs):
        raise NotImplementedError


class ObjectType(object):
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)

    def __eq__(self, other):
        try:
            ret = self.__dict__ == other.__dict__
        except AttributeError:
            # Numpy object types no longer have a __dict__ attribute in newer versions.
            ret = False
        return ret

    def create_vltype(self, dataset, name):
        if self.dtype == object:
            msg = 'Object/ragged arrays required a non-object datatype when writing to netCDF.'
            raise ValueError(msg)
        return dataset.createVLType(self.dtype, name)


class Variable(AbstractContainer, Attributes):
    """
    A variable contains data values. They may be masked and have attributes.

    :param str name: The variable's name (required). 
    :param value: The variable's data.
    :type value: :class:`numpy.ndarray` | :class:`numpy.ma.MaskedArray` | `sequence`
    :param dimensions: Dimensions for ``value``. The number of dimensions must match the dimension count of  ``value``
     (if provided). ``None`` is allowed for scalar or attribute container variables.
    :type dimensions: `sequence` of :class:`~ocgis.Dimension` | :class:`str`
    :param dtype: The variable's data type. If ``'auto'``, the data type will match the data type of ``value``. If the 
     data type does not match ``values``'s data type, ``value`` will be converted to match.
    :param mask: The variable's mask. If ``None`` and ``value`` is a :class:`numpy.ma.MaskedArray`, then the mask is 
     pulled from ``value``. Shape must be the same as ``value``. Data type is cast to ``bool``.
    :type mask: :class:`numpy.ndarray` | `sequence`
    :param attrs: See :class:`~ocgis.variable.attributes.Attributes`. 
    :param int fill_value: The fill value to use when hardening the mask. If ``'auto'``, this will be determined
     automatically from a masked array or the data type.
    :param str units: Units for the variable's data. If ``'auto'``, attempt to pull units from the variable's  
     ``attrs``. 
    :param parent: See :class:`~ocgis.variable.base.AbstractContainer`. 
    :param bounds: Bounds for the variable's data value. Mostly applicable for coordinate-type variables.
    :type bounds: :class:`~ocgis.Variable`
    :param is_empty: If ``True``, the variable is empty and has not value, mask, or meaning.
    :param source_name: See :class:`~ocgis.base.AbstractNamedObject`.
    :param uid: See :class:`~ocgis.base.AbstractNamedObject`.
    :param repeat_record: A value to repeat when the variable's :meth:`~ocgis.Variable.iter` method is called.
    :type repeat_record: `sequence`
    
    >>> repeat_record = [('i am', 'a repeater'), ('this is my value', 5)]
    
    **Example Code:**
    
    >>> # Create simple variable with a single dimension.
    >>> var = Variable(name='data', value=[1, 2, 3], dtype=float, dimensions='three')
    >>> assert var.dimensions[0].name == 'three'
    
    >>> # Create a variable using dimension objects.
    >>> from ocgis import Dimension
    >>> dim1 = Dimension('three', 3)
    >>> dim2 = Dimension('five', 5)
    >>> var = Variable(name='two_d', dimensions=[dim1, dim2], fill_value=4, dtype=int)
    >>> assert var.get_value().mean() == 4
    """
    _bounds_attribute_name = 'bounds'

    def __init__(self, name=None, value=None, dimensions=None, dtype='auto', mask=None, attrs=None, fill_value='auto',
                 units='auto', parent=None, bounds=None, is_empty=None, source_name=constants.UNINITIALIZED, uid=None,
                 repeat_record=None, dims=constants.UNINITIALIZED):
        #tdk: DOC: dims

        # For xarray compatibility, adjust the standard dimensions argument if "dims" is used in its place.
        if dims != constants.UNINITIALIZED:
            dimensions = dims

        if not is_empty:
            if name is None:
                raise ValueError('A variable name is required.')
            if value is not None and dimensions is None:
                msg = 'Variables with a value require dimensions. If this is a scalar variable, provide an empty ' \
                      'list or tuple.'
                raise ValueError(msg)

        self._is_init = True

        Attributes.__init__(self, attrs=attrs)

        self.repeat_record = repeat_record
        self._dimensions = None
        # Use to keep a copy of the dimensions if the variable is orphaned.
        self._dimensions_cache = constants.UNINITIALIZED
        self._value = None
        self._dtype = dtype
        self._mask = None
        self._bounds_name = None
        self._name_ugid = None

        self.dtype = dtype

        self._fill_value = fill_value

        AbstractContainer.__init__(self, name, parent=parent, source_name=source_name, uid=uid, is_empty=is_empty)

        # The variable will always be a member of the parent. Note this clobbers the name in the parent.
        self.parent[self.name] = self

        # Units on sourced variables may check for the presence of a parent. Units may be used by bounds, so set the
        # units here.
        if str(units) != 'auto':
            self.units = units

        self.set_value(value)
        self.set_dimensions(dimensions)
        if value is not None:
            update_unlimited_dimension_length(self.get_value(), self.dimensions)
        self.set_bounds(bounds)

        # Set the mask after setting the bounds. Bounds mask will be updated with the parent mask.
        if mask is not None:
            self.set_mask(mask)

        # Holds the global string length for the variable.
        self._string_max_length_global = None

        self._is_init = False

    def _getitem_main_(self, ret, slc):
        new_value = None
        new_mask = None

        if ret._value is not None:
            new_value = ret.get_value().__getitem__(slc)
        if ret._mask is not None:
            new_mask = ret._mask.__getitem__(slc)

        if new_value is not None:
            ret._value = new_value
        if new_mask is not None:
            ret._mask = new_mask

    def __len__(self):
        raise NotImplementedError("Use <object>.size")

    def __setitem__(self, slc, variable):
        """
        Set the this variable's value and mask to the other variable's value and mask in the index space defined by
        ``slc``.

        :param slc: The index space for setting data from ``variable``. If ``slc`` is a sequence, it must have the same
         length as the target variable's dimension count. If ``slc`` is a dictionary, there must be a key for each
         dimension name.
        :type slc: (:class:`slice`-like, ...) | dict(<str>=<`slice-like`>, ...)
        :param variable: The variable to use for setting values in the target.
        :type variable: :class:`~ocgis.Variable`
        """

        if isinstance(slc, dict):
            slc = [slc[ii] for ii in self.dimension_names]

        slc = get_formatted_slice(slc, self.ndim)
        self.get_value()[slc] = variable.get_value()

        variable_mask = variable.get_mask()
        if variable_mask is not None:
            new_mask = self.get_mask(create=True)
            new_mask[slc] = variable_mask

        if self.has_bounds:
            names_src = get_dimension_names(self.dimensions)
            names_dst = get_dimension_names(self.bounds.dimensions)
            slc = get_mapped_slice(slc, names_src, names_dst)
            self.bounds[slc] = variable.bounds

        if variable_mask is not None:
            self.set_mask(new_mask)

    @property
    def bounds(self):
        """
        :return: A bounds variable or ``None``
        :rtype: :class:`~ocgis.Variable` | ``None``
        """

        if self._bounds_name is None:
            ret = None
        else:
            ret = self.parent[self._bounds_name]
        return ret

    @property
    def cfunits(self):
        """
        :return: The CF units object representation.
        :rtype: :class:`cf_units.Units`
        """

        return get_units_object(self.units)

    @property
    def dtype(self):
        """
        Get or set the variable's data type. If ``'auto'``, this will be chosen automatically from the variable's
        ``numpy`` data type. Setting does not do any type conversion.

        :return: The data type for variable.
        :rtype: type
        """
        is_auto = is_auto_dtype(self._dtype)
        if is_auto:
            ret = self._get_dtype_()
        else:
            ret = self._dtype
        return ret

    def _get_dtype_(self):
        try:
            ret = self._value.dtype
            if ret == object:
                ret = ObjectType(object)
        except AttributeError:
            # Assume None.
            ret = None
        return ret

    @dtype.setter
    def dtype(self, value):
        self._dtype = value

    @property
    def dimensions_dict(self):
        """
        :return: Dimensions as a dictionary. Keys are the dimension names. Values are the dimension objects.
        :rtype: :class:`~collections.OrderedDict`
        """

        ret = OrderedDict()
        for d in self.dimensions:
            ret[d.name] = d
        return ret

    @property
    def dimension_names(self):
        """
        :return: A sequence of dimension names instead of objects.
        :rtype: :class:`tuple` of :class:`str`
        """
        if self._dimensions is None:
            ret = tuple()
        else:
            ret = self._dimensions
        return ret

    def _get_dimensions_(self):
        if self._dimensions is None:
            ret = tuple()
        else:
            if self.is_orphaned and self._dimensions_cache != constants.UNINITIALIZED:
                ret = self._dimensions_cache
            else:
                ret = tuple([self.parent.dimensions[name] for name in self._dimensions])
        return ret

    def set_dimensions(self, dimensions, force=False):
        """
        Set dimensions for the variable. These may be set to ``None``.

        :param dimensions: The new dimensions. Should be congruent with the target variable.
        :type dimensions: `sequence` of :class:`~ocgis.Dimension`
        :param bool force: If ``True``, clobber any existing dimensions on :attr:`~ocgis.Variable.parent`
        """

        if dimensions is not None:
            dimensions = list(get_iter(dimensions, dtype=(Dimension, str)))
            dimension_names = [None] * len(dimensions)
            for idx, dimension in enumerate(dimensions):
                try:
                    dimension_name = dimension.name
                    self.parent.add_dimension(dimension, force=force)
                except AttributeError:
                    dimension_name = dimension
                if dimension_name not in self.parent.dimensions:
                    self.parent.add_dimension(Dimension(dimension_name, self.shape[idx]))
                dimension_names[idx] = dimension_name
            if len(set(dimension_names)) != len(dimension_names):
                raise ValueError('Dimensions must be unique.')
            self._dimensions = tuple(dimension_names)
        else:
            self._dimensions = dimensions
        update_unlimited_dimension_length(self._value, self.dimensions)
        # Only update the bounds dimensions if this is not part of the variable initialization process. Bounds are
        # configured normally during initialization.
        if not self._is_init and self.has_bounds:
            if dimensions is None:
                bounds_dimensions = None
            else:
                bounds_dimensions = list(self.bounds.dimensions)
                bounds_dimensions[0:len(self.dimensions)] = self.dimensions
            self.bounds.set_dimensions(bounds_dimensions)

    @property
    def extent(self):
        """
        :return: The extent of the variable's *masked* value (minimum, maximum). Not applicable for all data types.
        :rtype: tuple
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """
        raise_if_empty(self)

        target = self._get_extent_target_()
        ret = target.compressed().min(), target.compressed().max()

        return ret

    def _get_extent_target_(self):
        if self.has_bounds:
            ret = self.bounds.get_masked_value()
        else:
            ret = self.get_masked_value()
        return ret

    @property
    def fill_value(self):
        """
        :return: The variable's fill value. If ``'auto'``, determin this automatically from ``numpy``.
        :rtype: :class:`int` or :class:`float`
        """

        if self._fill_value == 'auto':
            ret = self._get_fill_value_()
        else:
            ret = self._fill_value
        return ret

    def _get_fill_value_(self):
        return get_default_fill_value_from_dtype(self.dtype)

    @property
    def has_dimensions(self):
        """
        :return: ``True`` if the variable has dimensions.
        :rtype: bool
        """

        if self.dimensions is not None:
            if len(self.dimensions) > 0:
                ret = True
            else:
                ret = False
        else:
            ret = False
        return ret

    @property
    def dist(self):
        """
        :return: ``True`` if the variable has a distributed dimension.
        :rtype: bool
        """

        ret = False
        dimensions = self.dimensions
        if dimensions is not None:
            for d in dimensions:
                if d.dist:
                    ret = True
                    break
        return ret

    @property
    def has_mask(self):
        """
        :return: ``True`` if the variable has a mask.
        :rtype: bool
        """
        if self._mask is None:
            ret = False
        else:
            ret = True
        return ret

    @property
    def has_masked_values(self):
        """
        :return: ``True`` if any values are masked.
        :rtype: bool
        """

        if not self.has_mask:
            ret = False
        else:
            mask = self.get_mask()
            ret = mask.any()
        return ret

    @property
    def is_empty(self):
        """
        :return: ``is_empty`` set at initialization or ``True`` if any dimensions are empty and ``is_empty=None`` at
         initialization.
        :rtype: bool
        """
        if self._is_empty is None:
            ret = False
            if self.dist:
                for dim in self.dimensions:
                    if dim.is_empty:
                        ret = True
                        break
        else:
            ret = self._is_empty
        return ret

    @property
    def is_orphaned(self):
        """
        Variables are always part of collections. "Orphaning" is used to isolate variables to avoid infinite recursion
        when operating on variable collections.

        :return: ``True`` if the variable has no parent.
        :rtype: bool
        """
        return self._parent is None

    @property
    def is_string_object(self):
        """Return ``True`` if the variable contains string data."""
        ret = False
        dtype = self.dtype
        if is_string(dtype):
            if len(self.dimensions) > 0:
                archetype = self.get_value().flatten()[0]
                if is_string(type(archetype)):
                    ret = True
        return ret

    @property
    def ndim(self):
        """
        :return: The dimension count for the variable.
        :rtype: int
        """
        if self._dimensions is None:
            ret = 0
        else:
            ret = len(self._dimensions)
        return ret

    @property
    def resolution(self):
        """
        Resolution is computed using the differences between successive values up to
        :attr:`ocgis.constants.RESOLUTION_LIMIT`. Applicable mostly for spatial coordinate variables.

        :rtype: :class:`float` or :class:`int`
        :raises: ResolutionError
        """

        if not self.has_bounds and self.get_value().shape[0] < 2:
            msg = 'With no bounds and only a single coordinate, approximate resolution may not be determined.'
            raise ResolutionError(msg)
        elif self.has_bounds:
            res_bounds = self.bounds.get_value()[0:constants.RESOLUTION_LIMIT]
            res_array = res_bounds[:, 1] - res_bounds[:, 0]
            ret = np.abs(res_array).mean()
        else:
            res_array = np.diff(np.abs(self.get_value()[0:constants.RESOLUTION_LIMIT]))
            ret = np.abs(res_array).mean()
        return ret

    @property
    def shape(self):
        """
        :return: Shape of the variable.
        :rtype: tuple
        """
        return self._get_shape_()

    @property
    def size(self):
        """
        :return: Size of the variable (count of its elements)
        :rtype: int
        """
        ret = 1
        if len(self.shape) == 0:
            ret = 0
        else:
            for s in self.shape:
                ret *= s
        return ret

    @property
    def string_max_length_global(self):
        """
        Get the max string length. This only returns the private value. Call :meth:`~ocgis.Variable.set_string_max_length_global`
        to initialize the private value.

        This is the maximum length of the strings contained in the object across the current :class:`ocgis.OcgVM`.

        :return: int
        """
        return self._string_max_length_global

    @property
    def ugid(self):
        """
        :return: unique identifier variable
        :rtype: :class:`~ocgis.Variable`
        """

        if self._name_ugid is None:
            return None
        else:
            return self.parent[self._name_ugid]

    def _get_shape_(self):
        return get_shape_from_variable(self)

    @property
    def units(self):
        """
        Get or set the units.

        :return: Units for the object.
        :rtype: str
        """
        return self._get_units_()

    @units.setter
    def units(self, value):
        self._set_units_(value)

    def _get_units_(self):
        return get_attribute_property(self, 'units')

    def _set_units_(self, value):
        if value is not None:
            value = str(value)
        set_attribute_property(self, 'units', value)
        if self.bounds is not None:
            set_attribute_property(self.bounds, 'units', value)

    @property
    def value(self):
        raise NotImplementedError('Use <object>.get_value()')

    @property
    def masked_value(self):
        raise NotImplementedError('Use <object>.get_masked_value')

    def _get_value_(self):
        if self.is_empty:
            ret = None
        else:
            dimensions = self.dimensions
            if len(dimensions) == 0:
                ret = None
            else:
                if has_unsized_dimension(dimensions):
                    msg = 'Value shapes for variables with unlimited and unsized dimensions are undetermined.'
                    raise ValueError(msg)
                elif len(dimensions) > 0:
                    ret = variable_get_zeros(dimensions, self.dtype, fill=self.fill_value)
        return ret

    def remove_value(self):
        """Remove the value on the variable. The variable's value will be re-allocated when it is retrieved again."""
        self._value = None

    def set_value(self, value, update_mask=False):
        """
        Set the variable value.

        :param value: :class:`numpy.ndarray` | `sequence`
        :param update_mask: See :class:`~ocgis.Variable.set_mask`
        """

        mask_to_set = None
        should_set_mask = False
        if value is not None:
            if isinstance(value, MaskedArray):
                should_set_mask = True
                try:
                    self._fill_value = value.fill_value
                except AttributeError:
                    # Use default fill values in this case.
                    if not isinstance(value, MaskedConstant):
                        raise
                mask = value.mask.copy()
                if np.isscalar(mask):
                    new_mask = np.zeros(value.shape, dtype=bool)
                    new_mask.fill(mask)
                    mask = new_mask
                mask_to_set = mask
                value = value.data

            if is_auto_dtype(self._dtype):
                desired_dtype = None
            else:
                desired_dtype = self._dtype

            if not isinstance(value, ndarray):
                if isinstance(value, Dimension):
                    raise ValueError('Value type not recognized: {}'.format(Dimension))
                array_type = desired_dtype
                if isinstance(array_type, ObjectType):
                    array_type = object
                value = np.array(value, dtype=array_type)
                if isinstance(desired_dtype, ObjectType):
                    if desired_dtype.dtype != object:
                        for idx in range(value.shape[0]):
                            value[idx] = np.array(value[idx], dtype=desired_dtype.dtype)
            if desired_dtype is not None and desired_dtype != value.dtype and value.dtype != object:
                try:
                    value = value.astype(desired_dtype, copy=False)
                except TypeError:
                    value = value.astype(desired_dtype)

        if not self._is_init:
            if value is None:
                if value is None and self.ndim > 0:
                    raise ValueError('Only dimensionless variables may set their value to None.')
            else:
                update_unlimited_dimension_length(value, self.dimensions)

        self._value = value
        if should_set_mask:
            self.set_mask(mask_to_set, update=update_mask)

    def to_xarray(self):
        """
        Convert the variable to a :class:`xarray.DataArray`. This *does not* traverse the parent's hierararchy. Use the
        conversion method on the variable's parent to convert all variables in the collection.

        :rtype: :class:`xarray.DataArray`
        """
        from xarray import DataArray

        # Always access a time variable's numeric data.
        if hasattr(self, 'value_numtime'):
            data = self.value_numtime.data
        else:
            # Make sure we are passing the masked data array when converting the underlying dataset.
            data = self.mv()

        # Collect the variable's dimensions.
        dims = [d.to_xarray() for d in self.dimensions]

        return DataArray(data=data, dims=dims, attrs=self.attrs, name=self.name)

    def copy(self):
        """
        :return: A shallow copy of the variable.
        :rtype: :class:`~ocgis.Variable`
        """

        if self._parent is None:
            ret = AbstractContainer.copy(self)
            ret.attrs = ret.attrs.copy()
            if ret._value is not None:
                ret._value = ret._value.view()
            if ret._mask is not None:
                ret._mask = ret._mask.view()
        else:
            ret = self.parent.copy()[self.name]
        return ret

    def deepcopy(self, eager=False):
        """
        :param bool eager: If ``True``, also deep copy the variable's :attr:`~ocgis.Variable.parent`.
        :return: A deep copy of the variable.
        :rtype: :class:`~ocgis.Variable`
        """

        deepcopied = self.copy()

        if eager:
            raise NotImplementedError
        else:
            with orphaned(deepcopied):
                deepcopied.__dict__ = deepcopy(deepcopied.__dict__)
            deepcopied.parent.add_variable(deepcopied, force=True)
            if deepcopied.has_bounds:
                deepcopied.set_bounds(deepcopied.bounds.deepcopy(), force=True)

        return deepcopied

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
            raise NoUnitsError(self.name)

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

        # Let the data type and fill value load from the value array.
        self._dtype = 'auto'
        self._fill_value = 'auto'

        if self.has_bounds:
            self.bounds.cfunits_conform(to_units, from_units=from_units)

    def convert_to_empty(self):
        """
        Convert this variable to an empty variable. This sets the value and mask to ``None``. Also sets the
        :attr:`~ocgis.Variable.parent` to empty.
        """

        if self.is_orphaned:
            self._mask = None
            self._value = None
            self._is_empty = True
        else:
            self.parent.convert_to_empty()

    def get_masked_value(self):
        """
        Return the variable's value as a masked array.

        :rtype: :class:`numpy.ma.MaskedArray`
        """
        if isinstance(self.dtype, ObjectType):
            dtype = object
        else:
            dtype = self.dtype
        ret = np.ma.array(self.get_value(), mask=self.get_mask(), dtype=dtype, fill_value=self.fill_value)
        return ret

    def set_extrapolated_bounds(self, name_variable, name_dimension):
        """
        Set the bounds variable using extrapolation.

        :param str name_variable: Name of the bounds variable.
        :param str name_dimension: Name for the bounds dimension.
        """

        if self.bounds is not None:
            raise BoundsAlreadyAvailableError
        if self.dimensions is None:
            raise DimensionsRequiredError('Dimensions are required on the bounded variable.')

        bounds_value = None
        if self.ndim == 1:
            if not self.is_empty:
                bounds_value = get_bounds_from_1d(self.get_value())
            bounds_dimension_size = 2
        else:
            # TODO: consider renaming this functions to get_bounds_from_2d.
            if not self.is_empty:
                bounds_value = get_extrapolated_corners_esmf(self.get_value())
                bounds_value = create_ocgis_corners_from_esmf_corners(bounds_value)
            bounds_dimension_size = 4

        dimensions = list(self.dimensions)
        dimensions.append(Dimension(name=name_dimension, size=bounds_dimension_size))

        var = self.__class__(name=name_variable, value=bounds_value, dimensions=dimensions)
        self.set_bounds(var)

    @property
    def has_allocated_mask(self):
        """
        :return: ``True`` if the mask is allocated.
        :rtype: bool
        """
        return self._mask is not None

    @property
    def has_allocated_value(self):
        """
        :return: ``True`` if the value is allocated.
        :rtype: bool
        """
        return self._value is not None

    @property
    def has_bounds(self):
        """
        :return: ``True`` if the variable has bounds.
        :rtype: bool
        """
        if not self.is_orphaned and self.bounds is not None:
            ret = True
        else:
            ret = False
        return ret

    def get_mask(self, create=False, check_value=False, eager=True):
        """
        :param bool create: If ``True``, create the mask if it does not exist.
        :param check_value: If ``True``, check the variable's value for values matching
         :attr:`~ocgis.Variable.fill_value`. Matching indices are set to ``True`` in the created mask.
        :return: An array of ``bool`` data type with shape matching :attr:`~ocgis.Variable.shape`.
        :rtype: :class:`numpy.ndarray`
        """
        if self.is_empty:
            ret = None
        else:
            ret = self._mask
            if ret is None and create:
                ret = np.zeros(self.shape, dtype=bool)
                if check_value:
                    fill_value = self.fill_value
                    if fill_value is not None:
                        is_equal = self.get_value() == fill_value
                        ret[is_equal] = True
                self.set_mask(ret)
        return ret

    def set_mask(self, mask, cascade=False, update=False):
        """
        Set the variable's mask.

        :param mask: A boolean array with shape matching :attr:`~ocgis.Variable.shape`.
        :type mask: :class:`numpy.ndarray` | `sequence`
        :param bool cascade: If ``True``, set the mask on variables in :attr:`~ocgis.Variable.parent` to match this
         mask. Only sets the masks along shared dimensions.
        :param bool update: If ``True``, update the existing mask using a logical `or` operation.
        """
        if mask is not None:
            mask = np.array(mask, dtype=bool)
            assert mask.shape == self.shape
        if update:
            if self.has_allocated_mask and mask is not None:
                mask = np.logical_or(mask, self._mask)
            else:
                # If the new mask is none and we are updating, then the appropriate mask is the one that currently
                # exists. A none mask is an array of falses.
                if mask is None:
                    mask = self._mask
        self._mask = mask
        if cascade:
            self.parent.set_mask(self, update=update)
        else:
            # Bounds will be updated if there is a parent. Otherwise, update the bounds directly.
            if self.has_bounds:
                set_mask_by_variable(self, self.bounds)

    def allocate_value(self, fill=None):
        """
        Allocate the value for the variable.

        :param fill: If ``None``, use :attr:`~ocgis.Variable.fill_value`.
        """
        if fill is None:
            fill = self.fill_value
        the_zeros = variable_get_zeros(self.dimensions, self.dtype, fill=fill)
        self.set_value(the_zeros)

    def create_metadata(self):
        """
        See :meth:`~ocgis.base.AbstractInterfaceObject.create_metadata`
        """
        root = AbstractNamedObject.create_metadata(self)
        root['dimensions'] = get_dimension_names(self.dimensions)
        root['attrs'] = self.attrs
        return root

    def create_ugid(self, name, start=1, is_current=True, **kwargs):
        """
        Create a unique identifier variable for the variable. The returned variable will have the same dimensions.

        :param str name: The name for the new global identifier variable.
        :param int start: Starting value for the unique identifier.
        :param bool is_current: If ``True`` (the default) set this variable using :meth:`~ocgis.Variable.set_ugid`.
        :param dict kwargs: Additional arguments to variable creation.
        :rtype: :class:`~ocgis.Variable`
        """

        if self.is_empty:
            value = None
        else:
            value = np.arange(start, start + self.size).reshape(self.shape)
        ret = Variable(name=name, value=value, dimensions=self.dimensions, is_empty=self.is_empty, **kwargs)
        if is_current:
            self.set_ugid(ret)
        return ret

    def create_ugid_global(self, name, start=1):
        """
        Same as :meth:`~ocgis.Variable.create_ugid` but collective across the current :class:`~ocgis.OcgVM`.

        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """

        raise_if_empty(self)

        sizes = vm.gather(self.size)
        if vm.rank == 0:
            for idx, n in enumerate(vm.ranks):
                if n == vm.rank:
                    rank_start = start
                else:
                    vm.comm.send(start, dest=n)
                start += sizes[idx]
        else:
            rank_start = vm.comm.recv(source=0)

        return self.create_ugid(name, start=rank_start)

    def extract(self, keep_bounds=True, clean_break=False):
        """
        Extract the variable from its collection.

        :param bool keep_bounds: If ``True``, maintain any bounds associated with the target variable.
        :param bool clean_break: If ``True``, remove the target from the containing collection entirely.
        :rtype: :class:`~ocgis.Variable`
        """

        ret = self.copy()
        if self.has_initialized_parent:
            to_keep = [self.name]
            if keep_bounds and self.has_bounds:
                to_keep.append(self.bounds.name)

            if clean_break:
                original_parent = self.parent
                new_parent = ret.parent
                to_pop_in_new = set(original_parent.keys()).difference(set(to_keep))
                for tk in to_keep:
                    original_parent.remove_variable(tk)
                for tp in to_pop_in_new:
                    new_parent.pop(tp)
            else:
                for var in list(ret.parent.values()):
                    if var.name not in to_keep:
                        ret.parent.pop(var.name)

        ret = ret.parent[self.name]

        # Remove any dimensions not associated with the extracted variable from its parent collection.
        dimensions_to_keep = list(get_dimension_names(ret.dimensions))
        if keep_bounds and ret.has_bounds:
            dimensions_to_keep += list(get_dimension_names(ret.bounds.dimensions))
        dimensions_to_keep = set(dimensions_to_keep)
        dimensions_remaining = set(self.parent.dimensions.keys())
        dimensions_to_pop = dimensions_remaining.difference(dimensions_to_keep)
        for d in dimensions_to_pop:
            ret.parent.dimensions.pop(d)

        return ret

    def get_between(self, lower, upper, return_indices=False, closed=False, use_bounds=True):
        """
        :param lower: The lower value.
        :param upper: The upper value.
        :param bool return_indices: If ``True``, also return the indices used to slice the variable.
        :param bool closed: If ``False`` (the default), operate on the open interval (``>=``, ``<=``). If ``True``,
         operate on the closed interval (``>``, ``<``).
        :param bool use_bounds: If ``True``, use the bounds values for the between operation.
        :return: A sliced variable.
        :rtype: :class:`~ocgis.Variable`
        """

        assert lower <= upper

        # Determine if data bounds are contiguous (if bounds exists for the data). Bounds must also have more than one
        # row.
        is_contiguous = False
        if self.has_bounds:
            bounds_value = self.bounds.get_value()
            try:
                if len(set(bounds_value[0, :]).intersection(set(bounds_value[1, :]))) > 0:
                    is_contiguous = True
            except IndexError:
                # There is likely not a second row.
                if bounds_value.shape[0] == 1:
                    pass
                else:
                    raise

        # Subset operation when bounds are not present.
        if not self.has_bounds or not use_bounds:
            value = self.get_value()
            # Allow single-valued dimensions to be subset.
            if self.ndim == 0:
                value = np.array([value])
            if closed:
                select = np.logical_and(value > lower, value < upper)
            else:
                select = np.logical_and(value >= lower, value <= upper)
        # Subset operation in the presence of bounds.
        else:
            # Determine which bound column contains the minimum.
            if bounds_value[0, 0] <= bounds_value[0, 1]:
                lower_index = 0
                upper_index = 1
            else:
                lower_index = 1
                upper_index = 0
            # Reference the minimum and maximum bounds.
            bounds_min = bounds_value[:, lower_index]
            bounds_max = bounds_value[:, upper_index]

            # If closed is True, then we are working on a closed interval and are not concerned if the values at the
            # bounds are equivalent. It does not matter if the bounds are contiguous.
            if closed:
                select_lower = np.logical_or(bounds_min > lower, bounds_max > lower)
                select_upper = np.logical_or(bounds_min < upper, bounds_max < upper)
            else:
                # If the bounds are contiguous, then preference is given to the lower bound to avoid duplicate
                # containers (contiguous bounds share a coordinate).
                if is_contiguous:
                    select_lower = np.logical_or(bounds_min >= lower, bounds_max > lower)
                    select_upper = np.logical_or(bounds_min <= upper, bounds_max < upper)
                else:
                    select_lower = np.logical_or(bounds_min >= lower, bounds_max >= lower)
                    select_upper = np.logical_or(bounds_min <= upper, bounds_max <= upper)
            select = np.logical_and(select_lower, select_upper)

        if not select.any():
            raise EmptySubsetError(origin=self.name)

        # Allow single-valued dimensions to be subset.
        if self.ndim == 0:
            if select[0]:
                ret = self.copy()
            else:
                raise EmptySubsetError(origin=self.name)
        else:
            slices = np.ma.clump_masked(np.ma.array(select, mask=select, dtype=bool))
            if len(slices) == 1:
                select = slices[0]
            ret = self[select]

        if return_indices:
            indices = np.arange(select.shape[0])
            ret = (ret, indices[select])

        return ret

    def get_distributed_slice(self, slc):
        """
        Slice a distributed variable. Returned variable may be empty.

        :param slc: The slice indices. The length of ``slc`` must match the number of variable dimensions.
        :rtype: :class:`~ocgis.Variable`
        """

        raise_if_empty(self)

        slc = get_formatted_slice(slc, self.ndim)
        new_dimensions = [None] * self.ndim
        dimensions = self.dimensions
        for idx in range(self.ndim):
            new_dimensions[idx] = dimensions[idx].get_distributed_slice(slc[idx])

        is_or_will_be_empty = self.is_empty or any([nd.is_empty for nd in new_dimensions])

        if is_or_will_be_empty:
            ret = self.copy()
            ret.convert_to_empty()
        else:
            slc = get_formatted_slice(slc, self.ndim)
            local_slc = [slice(None)] * self.ndim
            for idx in range(self.ndim):
                slc_idx = slc[idx]
                if isinstance(slc_idx, slice):
                    if slc_idx != slice(None):
                        local_slc_args = get_global_to_local_slice([slc_idx.start, slc_idx.stop],
                                                                   dimensions[idx].bounds_local)
                        local_slc[idx] = slice(*local_slc_args)
                else:
                    # Allow for fancy slicing.
                    if slc_idx.dtype == bool:
                        f = arange_from_bool_ndarray(slc_idx)
                        # The new arange is empty because the slice is boolean with all False. Convert the returned
                        # object to empty.
                        if f.shape[0] == 0:
                            is_or_will_be_empty = True
                            ret = self.copy()
                            ret.convert_to_empty()
                            break
                        local_slc[idx] = f
                    else:
                        local_slc[idx] = slc_idx
            if not is_or_will_be_empty:
                ret = self[local_slc]
                ret.set_dimensions(new_dimensions, force=True)

        return ret

    def get_report(self):
        """
        :returns: A sequence of strings suitable for printing.
        :rtype: list[str, ...]
        """
        lines = ['Name = {0}'.format(self.name),
                 'Count = {0}'.format(self.get_value().shape[0]),
                 'Has Bounds = {0}'.format(self.has_bounds),
                 'Data Type = {0}'.format(self.dtype)]
        return lines

    def get_scatter_slices(self, splits):
        slices = create_nd_slices(splits, self.shape)
        return slices

    def get_value(self):
        """
        :return: The data values associated with the variable.
        :rtype: :class:`numpy.ndarray`
        """
        if self._value is None:
            self._value = self._get_value_()
        return self._value

    def get_iter(self, **kwargs):
        """
        :param kwargs: See source.
        :return: A variable iterator object.
        :rtype: :class:`~ocgis.variable.iterator.Iterator`
        """

        add_bounds = kwargs.pop(KeywordArgument.ADD_BOUNDS, False)
        driver = kwargs.pop(KeywordArgument.DRIVER, None)
        repeaters = kwargs.pop(KeywordArgument.REPEATERS, None)

        formatter = kwargs.get('formatter')
        if formatter is None and driver is not None:
            formatter = driver.iterator_formatter
            kwargs['formatter'] = formatter

        if add_bounds and self.has_bounds:
            if self.ndim > 1:
                msg = 'Bounds iteration only supported for one-dimensional variables. Variable "{}" has {}.'
                msg = msg.format(self.name, self.ndim)
                raise ValueError(msg)
            else:
                bounds_value = self.bounds._get_iter_value_()
                min_index, max_index = get_bounds_order_1d(bounds_value)
                lb_name, ub_name = get_bounds_names_1d(self.name)
                lower_bounds = self.__class__(name=lb_name, value=bounds_value[:, min_index],
                                              dimensions=self.dimensions)
                upper_bounds = self.__class__(name=ub_name, value=bounds_value[:, max_index],
                                              dimensions=self.dimensions)
                followers = [lower_bounds, upper_bounds]
        else:
            followers = None

        if repeaters is None and self.repeat_record is not None:
            repeaters = self.repeat_record
        elif repeaters is not None and self.repeat_record is not None:
            repeaters = repeaters + self.repeat_record
        kwargs[KeywordArgument.REPEATERS] = repeaters

        kwargs[KeywordArgument.VALUE] = self._get_iter_value_()
        kwargs[KeywordArgument.FOLLOWERS] = followers
        kwargs[KeywordArgument.MASK] = self.get_mask()

        itr = Iterator(self, **kwargs)

        if self.has_bounds:
            for f in itr.followers:
                f.formatter = formatter

        return itr

    def iter_dict_slices(self, dimensions=None):
        """
        :param dimensions: Dimensions to iterate.
        :type dimensions: `sequence`
        :return: Yields dictionary slices in the form `{'<dimension name>': <integer index>}`.
        :rtype: dict
        """
        if dimensions is None:
            dimensions = self.dimensions
        dnames = get_dimension_names(dimensions)
        sizes = [len(self.parent.dimensions[dn]) for dn in dnames]
        for dct in iter_dict_slices(dnames, sizes):
            yield dct

    def join_string_value(self):
        """Join well-formed string values."""

        new_value = np.zeros(self.shape[0], dtype=object)
        value = self.get_value()
        for ii in range(self.shape[0]):
            curr = value[ii, :].tolist()
            curr = [c.decode() for c in curr]
            curr = ''.join(curr)
            new_value[ii] = curr
        return new_value

    def load(self, *args, **kwargs):
        """
        Allows variables to be fake-loaded in the case of mixed pure variables and sourced variables. Actual
        implementations is in :class:`~ocgis.variable.SourcedVariable`
        """

    def m(self, *args, **kwargs):
        """See :meth:`ocgis.Variable.get_mask`"""
        return self.get_mask(*args, **kwargs)

    def mv(self, *args, **kwargs):
        """See :meth:`ocgis.Variable.get_masked_value`"""
        return self.get_masked_value(*args, **kwargs)

    def reshape(self, *args):
        assert not self.has_bounds

        new_shape = [len(dimension) for dimension in get_iter(args[0], dtype=Dimension)]

        original_value = self.get_value()
        if self.has_mask:
            original_mask = self.get_mask()
        else:
            original_mask = None

        self.set_mask(None)
        self._value = None
        self.set_dimensions(None)

        if original_mask is not None:
            new_mask = original_mask.reshape(*new_shape)
        else:
            new_mask = None

        self.set_dimensions(args[0])
        self.set_value(original_value.reshape(*new_shape))
        self.set_mask(new_mask)

    def set_bounds(self, value, force=False, clobber_units=None):
        """
        Set the bounds variable.

        :param value: The variable containing bounds for the target.
        :type value: :class:`~ocgis.Variable`
        :param bool force: If ``True``, clobber the bounds if they exist in :attr:`~ocgis.Variable.parent`.
        :param bool clobber_units: If ``True``, clobber ``value.units`` to match ``self.units``. If ``None``, default to
         :attr:`ocgis.env.CLOBBER_UNITS_ON_BOUNDS`
        """

        if clobber_units is None:
            clobber_units = env.CLOBBER_UNITS_ON_BOUNDS

        bounds_attr_name = self._bounds_attribute_name
        parent = self.parent
        if value is None:
            if self._bounds_name is not None:
                parent.pop(self._bounds_name)
                self.attrs.pop(bounds_attr_name, None)
            self._bounds_name = None
        else:
            self._bounds_name = value.name
            self.attrs[bounds_attr_name] = value.name
            parent.add_variable(value, force=force)
            # Do not naively set the units as it may insert a None into the attributes dictionary.
            if clobber_units and self.units is not None:
                value.units = self.units

            # This will synchronize the bounds mask with the variable's mask.
            if not self.is_empty:
                if self.has_allocated_value:
                    self.set_mask(self.get_mask())

        # Attempt to update the dimension map for bounds designation (variable collection parents do not have a
        # dimension map).
        if not self.is_empty and hasattr(parent, KeywordArgument.DIMENSION_MAP):
            dmap = parent.dimension_map
            dkey = dmap.inquire_is_xyz(self)
            if dkey is not None:
                dmap.set_bounds(dkey, value)

    def set_string_max_length_global(self, value=None):
        """
        See :attr:`~ocgis.Variable.string_max_length_global`.

        Call is collective across the current :class:`~ocgis.OcgVM`.
        """

        if value is None and self.is_string_object:
            local_max = max([len(e) for e in self.get_value()])
            rank_maxes = vm.gather(local_max)
            if vm.rank == 0:
                res = max(rank_maxes)
            else:
                res = None
            self._string_max_length_global = vm.bcast(res)
        elif value is not None:
            self._string_max_length_global = value
        else:
            pass

    def set_ugid(self, variable, attr_link_name=None):
        """
        Set the unique identifier for the variable.

        :param variable: The unique identifier variable.
        :type variable: :class:`~ocgis.Variable` | ``None``
        :param str attr_link_name: If provided, set an attribute with this name on the current variable with a value of
         ``variable``'s name.
        """

        if variable is None:
            self._name_ugid = None
            if attr_link_name is not None:
                self.attrs.pop(attr_link_name, None)
        else:
            self.parent.add_variable(variable, force=True)
            self._name_ugid = variable.name
            if attr_link_name is not None:
                self.attrs[attr_link_name] = variable.name

    def v(self):
        """See :meth:`ocgis.Variable.get_value`"""
        return self.get_value()

    def write(self, *args, **kwargs):
        """
        Write the variable object using the provided driver.

        :keyword driver: ``(='netcdf-cf')`` The driver for variable writing. Not all drivers support writing single
         variables.
        :param args: Arguments to the driver's ``write_variable`` call.
        :param kwargs: Keyword arguments to driver's ``write_variable`` call.
        """

        from ocgis.driver.nc import DriverNetcdf
        from ocgis.driver.registry import get_driver_class

        driver = kwargs.pop('driver', DriverNetcdf)
        driver = get_driver_class(driver, default=driver)
        args = list(args)
        args.insert(0, self)
        driver.write_variable(*args, **kwargs)

    def _as_record_(self, add_bounds=True, formatter=None, pytypes=False, allow_masked=True, pytype_primitives=False,
                    clobber_masked=True, bounds_names=None):
        # TODO: Remove this method. It has no known usages.
        if self.is_empty:
            return {}

        name = self.name
        if self.ndim == 0:
            ret = OrderedDict(([HeaderName.DATASET_IDENTIFER, self.parent.uid], [name, None]))
        else:
            assert self.shape[0] == 1
            value = self._get_iter_value_().flatten()[0]
            if self.has_mask:
                mask = self.get_mask()[0]
            else:
                mask = False
            if mask:
                if not allow_masked:
                    raise MaskedDataFound
                if clobber_masked:
                    value = self.fill_value
                else:
                    value = self.get_value().flatten()[0]
            if pytypes:
                value = np.array(value).tolist()
            ret = OrderedDict(([HeaderName.DATASET_IDENTIFER, self.parent.uid], [name, value]))

            if self.has_bounds and add_bounds:
                name_bounds = self.bounds.name
                values = self.bounds._get_iter_value_().flatten()
                if mask:
                    values = [self.fill_value] * len(values)
                for index, v in enumerate(values):
                    if pytypes:
                        v = np.array(v).tolist()
                    if bounds_names is None:
                        ret['{}_{}'.format(index, name_bounds)] = v
                    else:
                        ret[bounds_names[index]] = v

            if formatter:
                for k, v in list(ret.items()):
                    ret[k] = formatter(v)

            # Add the repeat record. Assume this is formatted appropriately by the client.
            if self.repeat_record is not None:
                ret.update(self.repeat_record)

        return ret

    def _get_is_empty_(self):
        return self.parent.is_empty

    def _get_iter_value_(self):
        return self.get_value()

    def _get_to_conform_value_(self):
        return self.get_masked_value()

    def _initialize_parent_(self, *args, **kwargs):
        self._parent = self._get_parent_class_()(*args, **kwargs)

    def _set_to_conform_value_(self, value):
        self.set_value(value)


class SourcedVariable(Variable):
    def __init__(self, *args, **kwargs):
        """
        Like a variable but loads its value and metadata from a source request dataset. Full variable functionality is 
        maintained for convenience. Generally, it is a good idea to only provide ``name` and ``request_dataset`` to 
        avoid conflicts.
        
        .. note:: Accepts all parameters to :class:`~ocgis.Variable`.

        Additional arguments and/or keyword arguments are:
        
        :keyword request_dataset: (``=None``) The request dataset containing the variable's source information.
        :type request_dataset: :class`ocgis.RequestDataset`
        :keyword bool protected: (``=False``) If ``True``, attempting to access the variable's value from source will
         raise a :class:`ocgis.exc.PayloadProtectedError` exception. Set `<object>.payload = False` to disable this.
         Useful to ensure the variables payload data is untouched through a series of operations.
        :keyword bool should_init_from_source: (``=True``) Allows a sourced variable to ignore any from-file operations
         and behave as a normal variable. This is used by some subclasses.
        """

        # If True, initialize from source. If False, assume all source data is passed during initialization.
        should_init_from_source = kwargs.pop('should_init_from_source', True)

        # Flag to indicate if value has been loaded from source. This allows the value to be set to None and not have a
        # reload from source.
        self._has_initialized_value = False
        self.protected = kwargs.pop('protected', False)
        self._request_dataset = kwargs.pop('request_dataset', None)
        kwargs['attrs'] = kwargs.get('attrs') or OrderedDict()
        bounds = kwargs.pop('bounds', None)
        super(SourcedVariable, self).__init__(*args, **kwargs)

        if should_init_from_source:
            init_from_source(self)

        if bounds is not None:
            self.set_bounds(bounds, force=True)

    def get_iter(self, **kwargs):
        ret = super(SourcedVariable, self).get_iter(**kwargs)

        if self._request_dataset is not None:
            repeaters = ret.repeaters
            did_record = (HeaderName.DATASET_IDENTIFER, self._request_dataset.uid)
            if repeaters is None:
                ret.repeaters = [did_record]
            else:
                repeaters.insert(did_record)

        return ret

    def get_mask(self, *args, **kwargs):
        if self._is_empty:
            return None

        eager = kwargs.pop(KeywordArgument.EAGER, True)
        # When load from disk, the current mask and from disk masked should be merged with a logical OR.
        if eager:
            self.load()
            current_mask = self._mask
            self.set_mask(None)
            from_source_mask = self.get_mask(eager=False)
            if current_mask is None:
                if from_source_mask is None:
                    new_mask = None
                else:
                    new_mask = from_source_mask
            else:
                if from_source_mask is not None:
                    new_mask = np.logical_or(current_mask, from_source_mask)
                else:
                    new_mask = current_mask
            self.set_mask(new_mask)
        return super(SourcedVariable, self).get_mask(*args, **kwargs)

    def load(self, cascade=False):
        """Load all variable data from source.

        :param bool cascade: If ``False``, only load this variable's data form source. If ``True``, load all data from
         source including any variables on its parent object.
        """

        # Only load the value if it has not been initialized and it is None.
        if not self._has_initialized_value:
            self._get_value_()
        if cascade and self.parent is not None:
            for var in list(self.parent.values()):
                var.load()

    def _get_value_(self):
        if not self.is_empty and self._value is None and not self._has_initialized_value and self._request_dataset.uri is not None:
            if self.protected:
                raise PayloadProtectedError(self.name)
            self._request_dataset.driver.init_variable_value(self)
            ret = self._value
            self._has_initialized_value = True
        else:
            ret = super(SourcedVariable, self)._get_value_()
        return ret


class VariableCollection(AbstractCollection, AbstractContainer, Attributes):
    """
    Variable collections behave like Python dictionaries. The keys are variable names and values are variable  objects. 
    A variable collection may have a parent and children (groups). Variable collections may be sliced using a 
    dictionary.

    :param str name: The collection's name.
    :param variables: Initial set of variables used to initialize the collection.
    :type variables: `sequence` of :class:`~ocgis.Variable`
    :param attrs: See :class:`~ocgis.variable.attributes.Attributes`.
    :param parent: The parent collection.
    :type parent: :class:`~ocgis.VariableCollection`
    :param children: A dictionary of child variable collections.
    :type children: dict
    
    >>> child_vc = VariableCollection()
    >>> children = {'child1': child_vc}
    
    :param aliases: See :class:`~ocgis.base.AbstractNamedObject`.
    :param tags: Tags are used to group variables (data variables for example).
    :type tags: dict
    
    >>> tags = {'special_variables': ['teddy', 'unicorn']}
    
    :param source_name: See :class:`~ocgis.base.AbstractNamedObject`.
    :param uid: See :class:`~ocgis.base.AbstractNamedObject`.
    :param is_empty: If ``True``, this is an empty collection.
    :param bool force: If ``True``, clobber any names that already exist in the collection.
    :param initial_data: See :class:`ocgis.collection.base.AbstractCollection`.
    :param groups: Alias for ``children``.
    :param dict dimensions: A dictionary of dimension objects. The keys are the dimension names. The values are :class:`~ocgis.Dimension`
     objects.
    """

    def __init__(self, name=None, variables=None, attrs=None, parent=None, children=None, aliases=None, tags=None,
                 source_name=constants.UNINITIALIZED, uid=None, is_empty=None, force=False, groups=None,
                 initial_data=None, dimensions=None):
        if dimensions is None:
            dimensions = OrderedDict()
        self._dimensions = dimensions

        if children is None and groups is not None:
            children = groups
        self.children = children or OrderedDict()

        if tags is None:
            tags = OrderedDict()
        self._tags = tags

        AbstractCollection.__init__(self, initial_data=initial_data)
        Attributes.__init__(self, attrs)
        AbstractContainer.__init__(self, name, aliases=aliases, source_name=source_name, uid=uid, parent=parent,
                                   is_empty=is_empty)

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable, force=force)

    def __getitem__(self, item_or_slc):
        """
        :param item_or_slc: The string name of the variable to retrieve or a dictionary slice. A dictionary slice has
         dimension names for keys and the slice as values. A shallow copy of the variable collection is returned in
         the case of a slice.
        :type item_or_slc: :class:`str` | :class:`dict`
        :return: :class:`~ocgis.Variable` | :class:`~ocgis.VariableCollection`
        """
        if not isinstance(item_or_slc, dict):
            ret = AbstractCollection.__getitem__(self, item_or_slc)
        else:
            #tdk: HACK: this will have to be driver-based as well
            if is_xarray(self.first()):
                ret = self.copy()
                ret._storage = self._storage.isel(**item_or_slc)
            else:
                # Assume a dictionary slice.
                self_dimensions = self.dimensions
                if len(self_dimensions) == 0:
                    raise DimensionsRequiredError

                ret = self.copy()
                for dimension_name, slc in list(item_or_slc.items()):
                    ret.dimensions[dimension_name] = self_dimensions[dimension_name].__getitem__(slc)
                names = set(item_or_slc.keys())
                for k, v in list(ret.items()):
                    with orphaned(v):
                        if v.ndim > 0:
                            v_dimension_names = set(v.dimension_names)
                            if len(v_dimension_names.intersection(names)) > 0:
                                mapped_slc = [None] * len(v.dimension_names)
                                for idx, dname in enumerate(v.dimension_names):
                                    mapped_slc[idx] = item_or_slc.get(dname, slice(None))
                                v_sub = v.__getitem__(mapped_slc)
                            else:
                                v_sub = v.copy()
                        else:
                            v_sub = v.copy()
                    ret.add_variable(v_sub, force=True)

        return ret

    @property
    def groups(self):
        """Alias for :attr:`~ocgis.VariableCollection.children`."""
        return self.children

    @property
    def shapes(self):
        """
        :return: A dictionary of variable shapes.
        :rtype: :class:`~collections.OrderedDict`
        """
        return OrderedDict([[k, v.shape] for k, v in list(self.items()) if not is_crs_variable(v)])

    @property
    def tags(self):
        raise NotImplementedError

    def add_child(self, child, force=False):
        """
        Add a child variable collection to the current variable collection.
        
        :param child: Child variable collection to add.
        :type child: :class:`~ocgis.VariableCollection`
        :param bool force: If ``True``, clobber any existing children with the same name.
        :raises: ValueError
        """
        if child.name in self.children and not force:
            raise ValueError("Child with name '{}' already in parent with name '{}'.".format(child.name, self.name))
        child.parent = self
        self.children[child.name] = child

    def add_dimension(self, dimension, force=False, check_src_idx=True):
        """
        Add a dimension to the variable collection.
        
        :param dimension: The dimension to add. Will raise an exception if the dimension name is found in the collection
         and the dimensions are not equal.
        :type dimension: :class:`~ocgis.Dimension`
        :param bool force: If ``True``, clobber any dimensions with the same name.
        :param bool check_src_idx: If ``True``, assert dimension source indices are equal. Raise a dimension mismatch
         error if they are not.
        :raises: :class:`~ocgis.exc.DimensionMismatchError`
        """
        existing_dim = self.dimensions.get(dimension.name)
        if existing_dim is not None and not force:
            if not existing_dim.eq(dimension, check_src_idx=check_src_idx):
                raise DimensionMismatchError(dimension.name, self.name)
        else:
            self.dimensions[dimension.name] = dimension

    def add_group(self, *args, **kwargs):
        """Alias for :meth:`~ocgis.VariableCollection.add_child`."""

        self.add_child(*args, **kwargs)

    def add_variable(self, variable, force=False):
        """
        Add a variable to the variable collection.
        
        :param variable: The variable to add.
        :param bool force: If ``True``, clobber any variables in the collection with the same name.
        :type variable: :class:`~ocgis.Variable`
        :raises: :class:`~ocgis.exc.VariableInCollectionError`
        """

        try:
            if variable.is_orphaned:
                if not force and variable.name in self:
                    raise VariableInCollectionError(variable)
                self[variable.name] = variable
                variable.parent = self
            else:
                # Only check the source index if the incoming variable does not have an allocated value. The source index
                # will only be used to load the values.
                if variable.has_allocated_value:
                    check_src_idx = False
                else:
                    check_src_idx = True

                for dimension in list(variable.parent.dimensions.values()):
                    self.add_dimension(dimension, force=force, check_src_idx=check_src_idx)
                for var in list(variable.parent.values()):
                    var.parent = None
                    self.add_variable(var, force=force)
        except AttributeError:
            self._storage.update({variable.name: variable})

    def append_to_tags(self, tag, to_append, create=True):
        """
        Append a variable name to a tag.
        
        :param str tag: The tag name. 
        :param to_append: The variable or variable name to append to the tag.
        :type to_append: :class:`str` | :class:`~ocgis.Variable`
        :param bool create: If ``True``, create the tag if it does not exist. 
        :raises: ValueError
        """

        to_append = get_variable_names(to_append)
        names = list(self.get_by_tag(tag, create=create, names_only=True))

        for t in to_append:
            if t in names:
                raise ValueError('"{}" already in tag "{}".'.format(t, tag))
            else:
                names.append(t)

        self._tags[tag] = names

    def convert_to_empty(self):
        """Convert the variable collection to an empty collection. This will convert every variable to empty."""

        for v in list(self.values()):
            with orphaned(v):
                v.convert_to_empty()

    def copy(self):
        """
        :return: A shallow copy of the variable collection. Member variables and dimensions are also shallow copied.
        :rtype: :class:`~ocgis.VariableCollection`
        """

        ret = AbstractCollection.copy(self)
        ret._tags = deepcopy(self._tags)
        if not is_xarray(self._storage):
            ret._dimensions = ret._dimensions.copy()
            for v in list(ret.values()):
                with orphaned(v):
                    ret[v.name] = v.copy()
                ret[v.name].parent = ret
            ret.children = ret.children.copy()
        return ret

    def create_metadata(self):
        """
        See :meth:`~ocgis.base.AbstractInterfaceObject.create_metadata`
        """
        root = AbstractNamedObject.create_metadata(self)
        variables = OrderedDict()
        for k, v in self.items():
            variables[k] = v.create_metadata()
        dimensions = OrderedDict()
        for k, v in self.dimensions:
            dimensions[k] = v.create_metadata()
        root['variables'] = variables
        root['dimensions'] = dimensions
        return root

    def create_tag(self, tag):
        """
        Create a tag.
        
        :param str tag: The tag name.
        :raises: ValueError
        """

        if tag in self._tags:
            raise ValueError('Tag "{}" already exists.'.format(tag))
        else:
            self._tags[tag] = []

    def find_by_attribute(self, key=None, value=None, pred=None):
        """
        Find a variable by searching attributes.

        :param str key: The attribute key. If ``None``, check all attribute values.
        :param value: The value to match. Takes precedence over ``pred``.
        :type value: <varying>
        :param pred: A function accepting the attribute value associated with ``key``. If ``pred`` returns ``True``,
         the variable matches.
        :type pred: function
        :rtype: tuple of :class:`~ocgis.Variable`
        :raises: ValueError
        """

        ret = []
        for v in self.values():
            if key is None or key in v.attrs:
                if key is None:
                    keys = v.attrs.keys()
                else:
                    keys = [key]
                for k in keys:
                    attr_value = v.attrs[k]
                    if value is not None:
                        match = attr_value == value
                    elif pred is not None:
                        match = pred(attr_value)
                    else:
                        raise ValueError("Either 'value' or 'pred' must be defined (not None).")
                    if match:
                        ret.append(v)
        return tuple(ret)

    def get_by_tag(self, tag, create=False, strict=False, names_only=False):
        """
        Tuple of variable objects that have the ``tag``.
        
        :param str tag: The tag to retrieve.
        :param bool create: If ``True``, create the tag if it does not exist.
        :param bool strict: If ``True``, raise exception if variable name is not found in collection.
        :param bool names_only: If ``True``, return names and not variable objects.
        :rtype: tuple(:class:`ocgis.Variable`, ...)
        """

        if tag not in self._tags and create:
            if create:
                self.create_tag(tag)
            else:
                raise KeyError("Tag '{}' not found and 'create' is False.".format(tag))
        names = self._tags[tag]
        ret = []
        for n in names:
            if names_only:
                ret.append(n)
            else:
                try:
                    ret.append(self[n])
                except KeyError:
                    if strict:
                        raise
        ret = tuple(ret)
        return ret

    def get_mask(self, *args, **kwargs):
        super(VariableCollection, self).get_mask(*args, **kwargs)

    def groups_to_variable(self, **kwargs):
        """
        Convert the group identifier to a variable using the variable creation keyword arguments ``kwargs``. The
        dimension of the new variable will be used to stack other variables sharing that dimension along that dimension.

        :param kwargs: See :class:`ocgis.Variable`
        :rtype: :class:`ocgis.VariableCollection`
        """

        kwargs = kwargs.copy()

        new_size = len(self.groups)
        new_value = np.array(list(self.groups.keys()))

        kwargs['value'] = new_value
        new_var = Variable(**kwargs)

        ret = dict_first(self.children.values()).copy()

        variables_to_stack = []
        for child in self.children.values():
            for var in child.iter_variables_by_dimensions(new_var.dimensions[0]):
                variables_to_stack.append(var)
            break

        for var in variables_to_stack:
            ret.remove_variable(var.name)

            new_stack_dims = OrderedDict()
            for dim in var.dimensions:
                new_stack_dims[dim.name] = len(dim)
            new_stack_dims[new_var.dimension_names[0]] = new_size

            for k, v in new_stack_dims.items():
                new_stack_dims[k] = Dimension(name=k, size=v)
            new_stack_var = Variable(name=var.name, dimensions=new_stack_dims.values(), dtype=var.dtype)
            new_stack_var.allocate_value()

            for ii, src_vc in enumerate(self.children.values()):
                slc = {new_var.dimensions[0].name: ii}
                src_var = src_vc[var.name]
                fill_ref = new_stack_var[slc]
                fill_ref.v()[:] = src_var.v()

            ret.add_variable(new_stack_var)

        ret.add_variable(new_var)

        return ret

    def iter(self, **kwargs):
        """
        :return: Yield record dictionaries for variables in the collection.
        :rtype: dict
        """

        # TODO: Document kwargs.
        # TODO: Move implementation to function to clean up class code. Do this for Field.iter as well.

        if self.is_empty:
            raise StopIteration

        from ocgis.driver.registry import get_driver_class

        header_map = kwargs.pop(KeywordArgument.HEADER_MAP, None)
        strict = kwargs.pop(KeywordArgument.STRICT, False)
        melted = kwargs.get(KeywordArgument.MELTED)
        driver = kwargs.pop(KeywordArgument.DRIVER, None)
        if driver is not None:
            driver = get_driver_class(driver)
        geom = kwargs.pop(KeywordArgument.GEOM, None)
        variable = kwargs.pop(KeywordArgument.VARIABLE, None)
        followers = kwargs.pop(KeywordArgument.FOLLOWERS, None)

        if geom is None:
            geom_name = None
        else:
            geom_name = get_variable_names(geom)[0]

        if variable is None:
            possible = list(self.values())
            if geom_name is not None:
                possible = [p for p in possible if p.name != geom_name]
            variable = possible[0]
            kwargs[KeywordArgument.FOLLOWERS] = possible[1:]
        else:
            if not isinstance(variable, Iterator):
                variable = get_variables(variable, self)[0]
            if followers is not None:
                for ii, f in enumerate(followers):
                    if not isinstance(f, Iterator):
                        followers[ii] = get_variables(f, self)[0]
            kwargs[KeywordArgument.FOLLOWERS] = followers

        has_melted = False if melted is None else True

        itr = Iterator(variable, **kwargs)

        if driver is not None:
            itr.formatter = driver.iterator_formatter

        if header_map is None:
            header_map_keys = None
        else:
            header_map_keys = list(header_map.keys())

        for yld in itr:
            if geom_name is None:
                geom_value = None
            else:
                geom_value = yld.pop(geom_name)
            if header_map is not None:
                new_yld = OrderedDict()
                for k, v in list(header_map.items()):
                    try:
                        new_yld[v] = yld[k]
                    except KeyError:
                        # Do not enforce that all keys in the header map are present unless this is strict.
                        if strict:
                            raise
                if has_melted:
                    new_yld[HeaderName.VARIABLE] = yld[HeaderName.VARIABLE]
                    new_yld[HeaderName.VALUE] = yld[HeaderName.VALUE]
                if not strict:
                    for k, v in list(yld.items()):
                        if k not in header_map_keys:
                            new_yld[k] = v
                yld = new_yld

            # Remove the calculation key if it is present on a non-melted format.
            if not has_melted:
                if HeaderName.CALCULATION_KEY in yld:
                    yld.pop(HeaderName.CALCULATION_KEY)

            yld = geom_value, yld
            yield yld

    def iter_variables_by_dimensions(self, dimensions):
        """
        :param dimensions: Dimensions required to select a variable.
        :type dimensions: `sequence` of :class:`str` | `sequence` of :class:`~ocgis.Dimension`
        :return: Yield variables sharing ``dimensions``.
        :rtype: :class:`~ocgis.Variable`
        """
        names = get_dimension_names(dimensions)
        for var in list(self.values()):
            if len(set(var.dimension_names).intersection(names)) == len(names):
                yield var

    def load(self):
        """Load all variable values (payloads) from source. Here for compatibility with sourced variables."""

        for v in list(self.values()):
            v.load()

    @staticmethod
    def read(*args, **kwargs):
        """
        Read a variable collection from a request dataset.
        
        :param args: Arguments to :class:`~ocgis.RequestDataset`.
        :param kwargs: Keyword arguments to :class:`~ocgis.RequestDataset`.
        :rtype: :class:`~ocgis.VariableCollection`
        """

        from ocgis import RequestDataset
        rd = RequestDataset(*args, **kwargs)
        return rd.driver.create_raw_field()

    def remove_orphaned_dimensions(self, dimensions=None):
        """
        Remove dimensions from the collection that are not associated with a variable in the current collection.

        :param dimensions: A sequence of :class:`~ocgis.Dimension` objects or string dimension names to check. If ``None``,
         check all dimensions in the collection.
        """
        if dimensions is None:
            dimensions = self.dimensions.values()
        dimensions = get_dimension_names(dimensions)
        dims_to_pop = []
        for d in dimensions:
            found = False
            for var in self.values():
                if d in var.dimension_names:
                    found = True
                    break
            if not found:
                dims_to_pop.append(d)
        for d in dims_to_pop:
            self.dimensions.pop(d)

    def remove_variable(self, variable, remove_bounds=True):
        """
        Remove a variable from the collection. This removes the variable's bounds by default. Any orphaned dimensions
        are removed from the collection following variable removal.

        :param variable: The variable or variable name to remove from the collection.
        :type variable: :class:`~ocgis.Variable` | :class:`str`
        :param bool remove_bounds: If ``True`` (the default), remove the variable's bounds from the collection.
        """

        variable = get_variables(variable, self)[0]
        dims_to_check = list(get_dimension_names(variable.dims))
        remove_names = [variable.name]
        if remove_bounds and has_bounds(variable):
            remove_names.append(variable.bounds.name)
            dims_to_check.extend(list(get_dimension_names(variable.bounds.dims)))
            dims_to_check = set(dims_to_check)

        if not is_xarray(variable):
            ret = variable.extract(keep_bounds=remove_bounds, clean_break=False)
        else:
            ret = variable

        # Remove name from tags
        for rn in remove_names:
            for v in list(self._tags.values()):
                if rn in v:
                    v.remove(rn)
            try:
                self.pop(rn)
            except AttributeError:  # This is xarray.
                self._storage.drop(rn)

        # Check for orphaned dimensions.
        if not is_xarray(variable):
            self.remove_orphaned_dimensions(dimensions=dims_to_check)

        return ret

    def rename_dimension(self, old_name, new_name):
        """
        Rename a dimension on the variable collection in-place.

        :param str old_name: The dimension's original name.
        :param str new_name: The dimension's new name.
        """
        target_dimension = self.dimensions.pop(old_name)
        target_dimension.set_name(new_name)
        self.dimensions[new_name] = target_dimension
        for var in self.values():
            dimnames = var.dimension_names
            if old_name in dimnames:
                new_dimension_names = list(dimnames)
                replace_idx = new_dimension_names.index(old_name)
                new_dimension_names[replace_idx] = new_name
                var._dimensions = tuple(new_dimension_names)

    def set_mask(self, variable, exclude=None, update=False):
        """
        Set all variable masks to the mask on `variable`. See :meth:`~ocgis.Variable.set_mask` for a description of how
        this works on variables.
        
        :param variable: The variable having the source mask.
        :type variable: :class:`~ocgis.Variable`
        :param exclude: Variables to exclude from mask setting.
        :type exclude: `sequence` of :class:`~ocgis.Variable` | `sequence` of :class:`str`
        :param update: See :meth:`~ocgis.Variable.set_mask`.
        """

        if exclude is not None:
            exclude = get_variable_names(exclude)
        names_container = [d for d in get_dimension_names(variable.dims)]
        for k, v in list(self.items()):
            if exclude is not None and k in exclude:
                continue
            if variable.name != k and v.ndim > 0:
                names_variable = [d for d in get_dimension_names(v.dims)]
                slice_map = get_mapping_for_slice(names_container, names_variable)
                if len(slice_map) > 0:
                    set_mask_by_variable(variable, v, slice_map=slice_map, update=update)

    def strip(self):
        """Remove dimensions, variables, and children from the collection."""

        self._storage = OrderedDict()
        self._dimensions = OrderedDict()
        self.children = OrderedDict()

    def to_xarray(self, **kwargs):
        """
        Convert all the variables in the collection to an :class:`xarray.Dataset`.

        :param kwargs: Optional keyword arguments to pass to the dataset creation. ``data_vars`` and ``attrs`` are
         always overloaded by this method.
        :rtype: :class:`xarray.Dataset`
        """
        #tdk: DOC: array_kwargs
        from xarray import Dataset

        kwargs = kwargs.copy()
        array_kwargs = kwargs.pop('array_kwargs', {})

        data_vars = OrderedDict()
        # Convert each variable to data array.
        for v in self.values():
            if not is_xarray(v):
                data_vars[v.name] = v.to_xarray(**array_kwargs)
            else:
                data_vars[v.name] = v

        # Create the arguments for the dataset creation.
        kwargs['data_vars'] = data_vars
        kwargs['attrs'] = self.attrs

        return Dataset(**kwargs)

    def write(self, *args, **kwargs):
        """
        Write the variable collection to file.

        :keyword driver: (`=`:attr:`ocgis.constants.DriverKey.NETCDF`) The driver to use for writing.
        :param args: Arguments to the driver's :meth:`~ocgis.driver.base.AbstractDriver.write_variable_collection`
         method.
        :param kwargs: Keyword arguments to the driver's :meth:`~ocgis.driver.base.AbstractDriver.write_variable_collection`
         method.
        """

        from ocgis.driver.registry import get_driver_class
        driver = kwargs.pop(KeywordArgument.DRIVER, DriverKey.NETCDF)
        driver = get_driver_class(driver)
        args = list(args)
        args.insert(0, self)
        driver.write_variable_collection(*args, **kwargs)

    def _get_dimensions_(self):
        if is_xarray(self._storage):
            ret = self._storage.dims
        else:
            ret = self._dimensions
        return ret

    def _get_is_empty_(self):
        return is_empty_recursive(self)

    def _initialize_parent_(self, *args, **kwargs):
        # Endless recursion if a collection always initializes a parent.
        self._parent = None

    def _validate_(self):
        ids_not_equal = []
        lens_not_equal = []
        for v in self.values():
            if id(v.parent) != id(self):
                ids_not_equal.append(v.name)
            if len(v.parent) != len(self):
                lens_not_equal.append(v.name)
        if len(ids_not_equal) > 0 or len(lens_not_equal) > 0:
            msg = 'Issues with parent relationships: ids={}, lens={}'.format(ids_not_equal, lens_not_equal)
            raise AssertionError(msg)


def are_variable_and_dimensions_shape_equal(variable_value, dimensions):
    to_test = []
    vshape = variable_value.shape
    dshape = get_dimension_lengths(dimensions)

    if len(vshape) != len(dshape):
        ret = False
    else:
        is_unlimited = [d.is_unlimited for d in dimensions]
        for v, d, iu in zip(vshape, dshape, is_unlimited):
            if iu:
                to_append = True
            else:
                to_append = v == d
            to_test.append(to_append)
        ret = all(to_test)

    return ret


def get_bounds_names_1d(base_name):
    return 'lb_{}'.format(base_name), 'ub_{}'.format(base_name)


def create_typed_variable_from_data_model(string_name, data_model=None, **kwargs):
    """
    Create a variable with the appropriate integer or float data type depending on the input data model.

    :param string_name: See :meth:`~ocgis.environment.get_dtype`.
    :param data_model: See :meth:`~ocgis.environment.get_dtype`.
    :param dict kwargs: Keyword arguments for the creation of the variable.
    :rtype: :class:`~ocgis.Variable`
    """
    from ocgis import Variable

    kwargs = kwargs.copy()
    kwargs[KeywordArgument.DTYPE] = get_dtype(string_name, data_model=data_model)
    return Variable(**kwargs)


def get_bounds_order_1d(bounds):
    min_index = np.argmin(bounds[0, :])
    max_index = np.abs(min_index - 1)
    return min_index, max_index


def get_attribute_property(variable, name):
    return variable.attrs.get(name)


def get_default_fill_value_from_dtype(dtype):
    if dtype is None or isinstance(dtype, ObjectType):
        ret = None
    else:
        ret = np.ma.array([1], dtype=dtype).get_fill_value()
        ret = np.array([ret], dtype=dtype)[0]
    return ret


def get_dimension_lengths(dimensions):
    ret = [len(d) for d in dimensions]
    return tuple(ret)


def get_dslice(dimensions, slc):
    ret = {}
    for d, s in zip(get_dimension_names(dimensions), slc):
        if not isinstance(s, slice):
            s = slice(*s)
        ret[d] = s
    return ret


def is_empty_recursive(target):
    if is_xarray(target) or is_xarray(dict_first(target.values())):
        ret = False
    else:
        if target._is_empty is None:
            ret = any([v.is_empty for v in list(target.values())])
            if not ret:
                for child in list(target.children.values()):
                    ret = is_empty_recursive(child)
        else:
            ret = target._is_empty
    return ret


def get_mapped_slice(slc_src, names_src, names_dst):
    ret = [slice(None)] * len(names_dst)
    for idx, name in enumerate(names_dst):
        try:
            idx_src = names_src.index(name)
        except ValueError:
            continue
        else:
            ret[idx] = slc_src[idx_src]
    return tuple(ret)


def get_mapping_for_slice(names_source, names_destination):
    to_map = set(names_source).intersection(names_destination)
    ret = []
    for name in to_map:
        ret.append([names_source.index(name), names_destination.index(name)])
    return ret


def get_shape_from_variable(variable):
    dimensions = variable._dimensions
    value = variable._value
    if dimensions is None and value is None:
        ret = tuple()
    elif dimensions is not None:
        ret = get_dimension_lengths(variable.dimensions)
    elif value is not None:
        ret = value.shape
    else:
        raise NotImplementedError()
    return ret


def get_slice_sequence_using_local_bounds(variable):
    ndim = variable.ndim
    ret = [None] * ndim
    for idx, dim in enumerate(variable.dimensions):
        lower, upper = dim.bounds_local
        ret[idx] = slice(lower, upper)
    return ret


def has_unlimited_dimension(dimensions):
    ret = False
    for d in dimensions:
        if d.is_unlimited:
            ret = True
            break
    return ret


def has_unsized_dimension(dimensions):
    ret = False
    for d in dimensions:
        if d.size is None and d.size_current is None:
            ret = True
            break
    return ret


def init_from_source(variable):
    request_dataset = variable._request_dataset
    if request_dataset is not None:
        request_dataset.driver.init_variable_from_source(variable)


def is_empty(varlike):
    if is_xarray(varlike):
        ret = False
    else:
        ret = varlike.is_empty
    return ret


def is_string(dtype):
    ret = False
    if dtype == object or np.dtype(dtype).kind in {'U', 'S'}:
        ret = True
    return ret


def set_attribute_property(variable, name, value):
    variable.attrs[name] = value


def set_bounds_mask_from_parent(mask, bounds):
    if mask.ndim == 1:
        mask_bounds = mask.reshape(-1, 1)
        mask_bounds = np.hstack((mask_bounds, mask_bounds))
    elif mask.ndim == 2:
        mask_bounds = np.zeros(list(mask.shape) + [4], dtype=bool)
        for idx_row, idx_col in itertools.product(list(range(mask.shape[0])), list(range(mask.shape[1]))):
            if mask[idx_row, idx_col]:
                mask_bounds[idx_row, idx_col, :] = True
    else:
        raise NotImplementedError(mask.ndim)
    bounds.set_mask(mask_bounds)


def set_mask_by_variable(source_variable, target_variable, slice_map=None, update=False):
    # Do not use an eager mask get. This will not load the data from source. Happens with netCDF data where
    # the mask is burned into the source data.
    # tdk: HACK: I guess this needs to be on a driver as well
    if is_xarray(source_variable):
        mask_source = source_variable.values
    else:
        mask_source = source_variable.get_mask(eager=False)
    if is_xarray(target_variable):
        mask_target = target_variable.values
        target_is_xarray = True  # tdk: FIX: this indirection will slow things down...
    else:
        mask_target = target_variable.get_mask(eager=False)
        target_is_xarray = False

    # If the source variable has no mask, there is no need to update the target.
    if mask_source is None:
        pass
    else:
        # This maps slice indices between source and destination.
        names_source = get_dimension_names(source_variable.dims)
        names_destination = get_dimension_names(target_variable.dims)
        if slice_map is None:
            slice_map = get_mapping_for_slice(names_source, names_destination)
        # If dimensions are equivalent, do not execute the loop.
        if names_source == names_destination:
            mask_target = mask_source
        else:
            # If the target mask is none, we will need to generate one.
            if mask_target is None:
                mask_target = np.zeros(target_variable.shape, dtype=bool)
            template = [slice(None)] * target_variable.ndim
            for slc in itertools.product(*[list(range(ii)) for ii in source_variable.shape]):
                # slc = [slice(s, s + 1) for s in slc]
                if mask_source[slc]:
                    for m in slice_map:
                        template[m[1]] = slc[m[0]]
                    if target_is_xarray:
                        mask_target[template] = np.nan
                    else:
                        mask_target[template] = True

        # tdk: FIX: this probably needs to be on a driver
        if not is_xarray(target_variable):
            target_variable.set_mask(mask_target, update=update)


def stack(targets, stack_dim):
    """
    Stack targets vertically using the stack dimension. For example, this function may be used to concatenate
    variables along the time dimension.

    * For variables, the stack dimension object, value, and mask on the new variable will be a deep copy.
    * For variables, the collection hierarchy is not traversed. Use the parent collections directly for the stack
      method.
    * For collections, the returned value is a copy of the first field with stacked variables as deep copies. If a
      variable's dimensions does not contain the stacked dimension, it is a returned as a copy.

    :param targets: List of variables, variable collections, or fields to stack vertically along the stack
     dimension.
    :type targets: [:class:`~ocgis.Variable` | :class:`~ocgis.VariableCollection` | :class:`~ocgis.Field`, ...]
    :param stack_dim: Dimension to use for vertical stacking.
    :type stack_dim: ``str`` | :class:`~ocgis.Dimension`
    :rtype: Same as input type to ``targets``.
    """

    # The archetype is used as a template for attributes and dimension iteration.
    arch = targets[0]

    raise_if_empty(arch)

    if isinstance(arch, VariableCollection):
        # Collection stacking will call the stack method recursively for each variable that shares a stack
        # dimension.
        to_stack = get_variable_names(arch.iter_variables_by_dimensions(stack_dim))
        # A shallow copy of the collection will be the return target.
        ret = arch.copy()
        # Stack each variable that shares the stack dimension.
        stacked = [stack([vc[varname] for vc in targets], stack_dim) for varname in to_stack]
        # Replace the stacked variables and dimension on the outgoing collection.
        for s in stacked:
            for dim in s.dimensions:
                ret.dimensions[dim.name] = dim
            ret.add_variable(s, force=True)  # Variable already exists in the outgoing collection so must be forced
    else:
        # Name of the stack dimension.
        stack_dim = get_dimension_names(stack_dim)[0]
        # Will hold new size for the stack dimension.
        new_size = 0

        # Accumulate the size of the new dimension by adding it size form the incoming variables.
        for var in targets:
            new_size += len(var.dimensions_dict[stack_dim])

        # The dimensions on stacked variables will be modified. Load any data from disk before modifying those
        # dimensions as the source relationships will be lost.
        for var in targets:
            var.load()

        # Construct the new dimensions for the stacked variable.
        new_dimensions = [None] * arch.ndim
        for ii, dim in enumerate(arch.dimensions):
            if dim.name == stack_dim:
                # If this is the stack dimension, we need to create a new dimension as the size will change.
                is_unlimited = dim.is_unlimited  # Maintain a dimension's unlimited state.
                if is_unlimited:
                    nd = Dimension(name=dim.name, size_current=new_size)
                else:
                    nd = Dimension(name=dim.name, size=new_size)
            else:
                nd = dim
            new_dimensions[ii] = nd

        # Construct the outgoing variable.
        new_var = Variable(name=arch.name, dtype=arch.dtype, dimensions=new_dimensions, attrs=arch.attrs,
                           fill_value=arch.fill_value)

        # Fill the outgoing variable with data from the original variables.
        for ii, var in enumerate(targets):
            if ii == 0:
                # Construct the template slice if this is the first iteration.
                slc = {dim.name: slice(0, len(dim)) for dim in var.dimensions}
            else:
                # Adjust the slice as we progress through the source variables.
                for dim in var.dimensions:
                    if dim.name == stack_dim:
                        prev_stop = slc[dim.name].stop
                        slc[dim.name] = slice(prev_stop, prev_stop + len(dim))
                        break
            # Set the data using variable set item capabilities.
            new_var[slc] = var

        ret = new_var

    return ret


def update_unlimited_dimension_length(variable_value, dimensions):
    """
    Update unlimited dimension length if present on the variable. Update only occurs if the variable's value is
    allocated.
    """
    if variable_value is not None:
        # Update any unlimited dimension length.
        if dimensions is not None:
            aq = are_variable_and_dimensions_shape_equal(variable_value, dimensions)
            if not aq:
                msg = "Variable and dimension shapes must be equal."
                raise ValueError(msg)
            for idx, d in enumerate(dimensions):
                if d.size is None:
                    d._size_current = variable_value.shape[idx]


def variable_get_zeros(dimensions, dtype, fill=None):
    new_shape = get_dimension_lengths(dimensions)
    ret = np.zeros(new_shape, dtype=dtype)
    if fill is not None:
        ret.fill(fill)
    return ret

