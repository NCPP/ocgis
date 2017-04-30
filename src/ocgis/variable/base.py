import abc
import itertools
from abc import abstractproperty, abstractmethod
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import six
from numpy.core.multiarray import ndarray
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant

from ocgis import constants
from ocgis.base import AbstractNamedObject, get_dimension_names, get_variable_names, get_variables, iter_dict_slices, \
    orphaned, raise_if_empty
from ocgis.collection.base import AbstractCollection
from ocgis.constants import HeaderName, KeywordArgument, DriverKey
from ocgis.exc import VariableInCollectionError, BoundsAlreadyAvailableError, EmptySubsetError, \
    ResolutionError, NoUnitsError, DimensionsRequiredError, DimensionMismatchError, MaskedDataFound
from ocgis.util.helpers import get_iter, get_formatted_slice, get_bounds_from_1d, get_extrapolated_corners_esmf, \
    get_ocgis_corners_from_esmf_corners, is_crs_variable
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

    :param parent: (``=None``) The parent collection for this container. A variable will always become a member of its
     parent.
    :type parent: :class:`ocgis.VariableCollection`
    """

    def __init__(self, name, aliases=None, source_name=constants.UNINITIALIZED, parent=None, uid=None):
        self._parent = parent

        if parent is None:
            self._initialize_parent_()

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

    @abstractproperty
    def dimensions(self):
        """
        :return: A dimension dictionary containing all dimensions on associated with variables in the collection.
        :rtype: :class:`~collections.OrderedDict`
        """
        pass

    @property
    def group(self):
        """
        :return: The group index in the parent/child hierarchy. Returns ``None`` if this collection is the head.
        :rtype: ``None`` | :class:`list` of :class:`str`
        """

        curr = self.parent
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
        
        :param name: See :class:`~ocgis.base.AbstractNamedObject`.
        """
        if self.name in self.parent:
            self.parent[name] = self.parent.pop(self.name)
        super(AbstractContainer, self).set_name(name, aliases=aliases)

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

    def _getitem_main_(self, ret, slc):
        """Perform major slicing operations in-place."""

    def _getitem_finalize_(self, ret, slc):
        """Finalize the returned sliced object in-place."""

    def _initialize_parent_(self):
        self._parent = VariableCollection()


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


# tdk: order methods
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
                 repeat_record=None):

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
        self._is_empty = is_empty
        self._bounds_name = None

        self.dtype = dtype

        self._fill_value = fill_value

        AbstractContainer.__init__(self, name, parent=parent, source_name=source_name, uid=uid)

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
        
        :param slc: The index space for setting data from ``variable``. ``slc`` must have the same length as the
         target variable's dimension count.
        :type slc: (:class:`slice`-like, ...)
        :param variable: The variable to use for setting values in the target.
        :type variable: :class:`~ocgis.Variable`
        """

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

    def set_bounds(self, value, force=False):
        """
        Set the bounds variable.
        
        :param value: The variable containing bounds for the target.
        :type value: :class:`~ocgis.Variable`
        :param bool force: If ``True``, clobber the bounds if they exist in :attr:`~ocgis.Variable.parent`. 
        """

        bounds_attr_name = self._bounds_attribute_name
        if value is None:
            if self._bounds_name is not None:
                self.parent.pop(self._bounds_name)
                self.attrs.pop(bounds_attr_name, None)
            self._bounds_name = None
        else:
            self._bounds_name = value.name
            self.attrs[bounds_attr_name] = value.name
            self.parent.add_variable(value, force=force)
            value.units = self.units

            # This will synchronize the bounds mask with the variable's mask.
            if not self.is_empty:
                if self.has_allocated_value:
                    self.set_mask(self.get_mask())

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
    def dimensions(self):
        """
        :return: A tuple of dimension objects. 
        :rtype: :class:`tuple` of :class:`~ocgis.Dimension`
        """
        return self._get_dimensions_()

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
        dtype = self.dtype
        if dtype is None or isinstance(dtype, ObjectType):
            ret = None
        else:
            ret = np.ma.array([1], dtype=dtype).get_fill_value()
            ret = np.array([ret], dtype=dtype)[0]
        return ret

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

        if not self._is_init and value is not None:
            update_unlimited_dimension_length(value, self.dimensions)

        self._value = value
        if should_set_mask:
            self.set_mask(mask_to_set, update=update_mask)

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
            self.set_mask(None)
            self.set_value(None)
            self._is_empty = True
        else:
            self.parent.convert_to_empty()

    def get_masked_value(self):
        """
        :return: The variable's value as a masked array.
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
                bounds_value = get_ocgis_corners_from_esmf_corners(bounds_value)
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
         :attr:`~ocgis.Variabe.fill_value`. Matching indices are set to ``True`` in the created mask.
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

    #tdk: remove me or document
    def as_record(self, add_bounds=True, formatter=None, pytypes=False, allow_masked=True, pytype_primitives=False,
                  clobber_masked=True, bounds_names=None):
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

    def extract(self, keep_bounds=True, clean_break=False):
        """
        Extract the variable from its collection.
        
        :param bool keep_bounds: If ``True``, maintain any bounds associated with the target variable. 
        :param bool clean_break: If ``True``, remove the target from the containing collection entirely. 
        :rtype: :class:`~ocgis.Variable`
        """
        if self.has_initialized_parent:
            to_keep = [self.name]
            if keep_bounds and self.has_bounds:
                to_keep.append(self.bounds.name)

            if clean_break:
                original_parent = self.parent
                new_parent = self.parent.copy()
                to_pop_in_new = set(original_parent.keys()).difference(set(to_keep))
                for tk in to_keep:
                    original_parent.remove_variable(tk)
                for tp in to_pop_in_new:
                    new_parent.pop(tp)
                self.parent = new_parent
            else:
                self.parent = self.parent.copy()
                for var in list(self.parent.values()):
                    if var.name not in to_keep:
                        self.parent.pop(var.name)

        return self.parent[self.name]

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
                if slc[idx] != slice(None):
                    local_slc_args = get_global_to_local_slice([slc[idx].start, slc[idx].stop],
                                                               dimensions[idx].bounds_local)
                    local_slc[idx] = slice(*local_slc_args)
            ret = self[local_slc]

        if not is_or_will_be_empty:
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

    # tdk: remove me?
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

    def _get_iter_value_(self):
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

    def _get_to_conform_value_(self):
        return self.get_masked_value()

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

        # Flag to indicate if value has been loaded from sourced. This allows the value to be set to None and not have a
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

    def set_value(self, value, **kwargs):
        # Allow value to be set to None. This will remove dimensions.
        if self._has_initialized_value and value is None:
            self._dimensions = None
        super(SourcedVariable, self).set_value(value, **kwargs)

    def _get_value_(self):
        if not self.is_empty and self._value is None and not self._has_initialized_value:
            self._request_dataset.driver.init_variable_value(self)
            ret = self._value
            self._has_initialized_value = True
        else:
            ret = super(SourcedVariable, self)._get_value_()
        return ret


# TODO: Variable collection should inherit from abstract container.
class VariableCollection(AbstractNamedObject, AbstractCollection, Attributes):
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
    :param driver: A driver contains format-specific data transformations.
    :type driver: :class:`~ocgis.driver.base.AbstractDriver`
    :param bool force: If ``True``, clobber any names that already exist in the collection.
    :param groups: Alias for ``children``.
    """

    def __init__(self, name=None, variables=None, attrs=None, parent=None, children=None, aliases=None, tags=None,
                 source_name=constants.UNINITIALIZED, uid=None, is_empty=None, driver=constants.DEFAULT_DRIVER,
                 force=False, groups=None):
        self._is_empty = is_empty
        self._dimensions = OrderedDict()
        self.parent = parent

        if children is None and groups is not None:
            children = groups
        self.children = children or OrderedDict()
        self._driver = driver

        if tags is None:
            tags = OrderedDict()
        self._tags = tags

        AbstractCollection.__init__(self)
        Attributes.__init__(self, attrs)
        AbstractNamedObject.__init__(self, name, aliases=aliases, source_name=source_name, uid=uid)

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
                            mapped_slc = [None] * len(v_dimension_names)
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
    def dimensions(self):
        """
        :return: A dimension dictionary containing all dimensions on associated with variables in the collection.
        :rtype: :class:`~collections.OrderedDict`
        """
        return self._dimensions

    @property
    def driver(self):
        """
        :return: Get the driver associated with the collection.
        :rtype: :class:`~ocgis.AbstractDriver`
        """
        from ocgis.driver.registry import get_driver_class
        return get_driver_class(self._driver)

    @property
    def group(self):
        """
        :return: The group index in the parent/child hierarchy. Returns ``None`` if this collection is the head.
        :rtype: ``None`` | :class:`list` of :class:`str`
        """
        if self.parent is None:
            ret = None
        else:
            curr = self.parent
            ret = [curr.name]
            while True:
                if curr.parent is None:
                    break
                else:
                    curr = curr.parent
                    ret.append(curr.name)
            ret.reverse()
        return ret

    @property
    def groups(self):
        """Alias for :attr:`~ocgis.VariableCollection.children`."""
        return self.children

    @property
    def is_empty(self):
        """
        :return: ``True`` if there is anything empty in the collection.
        :rtype: bool
        """
        return get_is_empty_recursive(self)

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

    def add_dimension(self, dimension, force=False):
        """
        Add a dimension to the variable collection.
        
        :param dimension: The dimension to add. Will raise an exception if the dimension name is found in the collection
         and the dimensions are not equal.
        :type dimension: :class:`~ocgis.Dimension`
        :param bool force: If ``True``, clobber any dimensions with the same name.
        :raises: :class:`~ocgis.exc.DimensionMismatchError`
        """
        existing_dim = self.dimensions.get(dimension.name)
        if existing_dim is not None and not force:
            if existing_dim != dimension:
                raise DimensionMismatchError(dimension.name, self.name)
        else:
            self.dimensions[dimension.name] = dimension

    def add_variable(self, variable, force=False):
        """
        Add a variable to the variable collection.
        
        :param variable: The variable to add.
        :param bool force: If ``True``, clobber any variables in the collection with the same name.
        :type variable: :class:`~ocgis.Variable`
        :raises: :class:`~ocgis.exc.VariableInCollectionError`
        """

        if variable.is_orphaned:
            if not force and variable.name in self:
                raise VariableInCollectionError(variable)
            self[variable.name] = variable
            variable.parent = self
        else:
            for dimension in list(variable.parent.dimensions.values()):
                self.add_dimension(dimension, force=force)
            for var in list(variable.parent.values()):
                var.parent = None
                self.add_variable(var, force=force)

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
        try:
            names = self.get_by_tag(tag)
        except KeyError:
            if create:
                self.create_tag(tag)
                names = self.get_by_tag(tag)
            else:
                raise
        names = list(get_variable_names(names))

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
        ret._dimensions = ret._dimensions.copy()
        for v in list(ret.values()):
            with orphaned(v):
                ret[v.name] = v.copy()
            ret[v.name].parent = ret
        ret.children = ret.children.copy()
        return ret

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

    def get_by_tag(self, tag, create=False, strict=False):
        """
        :param str tag: The tag to retrieve.
        :param bool create: If ``True``, create the tag if it does not exist.
        :param bool strict: If ``True``, raise exception if variable name is not found in collection.
        :return: Tuple of variable objects that have the ``tag``.
        :rtype: tuple(:class:`ocgis.Variable`, ...)
        """
        try:
            names = self._tags[tag]
        except KeyError:
            if create:
                self.create_tag(tag)
                names = self._tags[tag]
            else:
                raise
        ret = []
        for n in names:
            try:
                ret.append(self[n])
            except KeyError:
                if strict:
                    raise
        ret = tuple(ret)
        return ret

    def load(self):
        """Load all variable values (payloads) from source. Here for compatibility with sourced variables."""

        for v in list(self.values()):
            v.load()

    # tdk: move implementation to function (also for field)
    # TODO: Document kwargs.
    def iter(self, **kwargs):
        """
        :return: Yield record dictionaries for variables in the collection.
        :rtype: dict
        """

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
        return rd.driver.get_variable_collection()

    def remove_variable(self, variable, remove_bounds=True):
        """
        Remove a variable from the collection. This removes the variable's bounds by default. Any orphaned dimensions
        are removed from the collection following variable removal.
        
        :param variable: The variable or variable name to remove from the collection.
        :type variable: :class:`~ocgis.Variable` | :class:`str`
        :param bool remove_bounds: If ``True`` (the default), remove the variable's bounds from the collection.
        """

        variable = get_variables(variable, self)[0]

        variable = variable.extract()
        if remove_bounds and variable.has_bounds:
            self.remove_variable(variable.bounds)

        # Remove name from tags
        variable_name = variable.name
        for v in list(self._tags.values()):
            if variable_name in v:
                v.remove(variable_name)
        self.pop(variable_name)

        # Check for orphaned dimensions.
        dims_to_check = get_dimension_names(variable.dimensions)
        dims_to_pop = []
        for d in dims_to_check:
            found = False
            for var in self.values():
                if d in var.dimension_names:
                    found = True
                    break
            if not found:
                dims_to_pop.append(d)
        for d in dims_to_pop:
            self.dimensions.pop(d)

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
        names_container = [d.name for d in variable.dimensions]
        for k, v in list(self.items()):
            if exclude is not None and k in exclude:
                continue
            if variable.name != k and v.ndim > 0:
                names_variable = [d.name for d in v.dimensions]
                slice_map = get_mapping_for_slice(names_container, names_variable)
                if len(slice_map) > 0:
                    set_mask_by_variable(variable, v, slice_map=slice_map, update=update)

    def strip(self):
        """Remove dimensions, variables, and children from the collection."""

        self._storage = OrderedDict()
        self._dimensions = OrderedDict()
        self.children = OrderedDict()

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


def get_bounds_order_1d(bounds):
    min_index = np.argmin(bounds[0, :])
    max_index = np.abs(min_index - 1)
    return min_index, max_index


def get_attribute_property(variable, name):
    return variable.attrs.get(name)


def get_dimension_lengths(dimensions):
    ret = [len(d) for d in dimensions]
    return tuple(ret)


def get_dslice(dimensions, slc):
    return {d.name: s for d, s in zip(dimensions, slc)}


def get_is_empty_recursive(target):
    if target._is_empty is None:
        ret = any([v.is_empty for v in list(target.values())])
        if not ret:
            for child in list(target.children.values()):
                ret = get_is_empty_recursive(child)
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
    # Do not use an eager mask get. This will not load the data from source. Happens with netCDF data where the mask
    # is burned into the source data.
    mask_source = source_variable.get_mask(eager=False)
    mask_target = target_variable.get_mask(eager=False)

    # If the source variable has no mask, there is no need to update the target.
    if mask_source is None:
        pass
    else:
        # This maps slice indices between source and destination.
        names_source = get_dimension_names(source_variable.dimensions)
        names_destination = get_dimension_names(target_variable.dimensions)
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
                    mask_target[template] = True
        target_variable.set_mask(mask_target, update=update)


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
