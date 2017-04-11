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
    get_ocgis_corners_from_esmf_corners
from ocgis.util.units import get_units_object, get_conformed_units
from ocgis.variable.attributes import Attributes
from ocgis.variable.crs import CoordinateReferenceSystem
from ocgis.variable.dimension import Dimension
from ocgis.variable.iterator import Iterator
from ocgis.vmachine.mpi import create_nd_slices, get_global_to_local_slice


def handle_empty(func):
    def wrapped(*args, **kwargs):
        self = args[0]
        if self.is_empty:
            return self
        else:
            return func(*args, **kwargs)

    return wrapped


@six.add_metaclass(abc.ABCMeta)
class AbstractContainer(AbstractNamedObject):
    def __init__(self, name, aliases=None, source_name=constants.UNINITIALIZED, parent=None, uid=None):
        self._parent = parent

        if parent is None:
            self._initialize_parent_()

        super(AbstractContainer, self).__init__(name, aliases=aliases, source_name=source_name, uid=uid)

    def __getitem__(self, slc):
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
        pass

    @property
    def group(self):
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
        return self._parent is not None

    @property
    def parent(self):
        if self._parent is None:
            self._initialize_parent_()
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

    @abstractmethod
    def get_mask(self):
        """:rtype: :class:`numpy.ndarray`"""
        raise NotImplementedError

    @abstractmethod
    def set_mask(self, mask):
        raise NotImplementedError

    def set_name(self, name, aliases=None):
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


class Variable(AbstractContainer, Attributes):
    _bounds_attribute_name = 'bounds'

    def __init__(self, name=None, value=None, dimensions=None, dtype=None, mask=None, attrs=None, fill_value=None,
                 units='auto', parent=None, bounds=None, is_empty=None,
                 source_name=constants.UNINITIALIZED, uid=None, repeat_record=None):
        if not is_empty:
            if name is None:
                raise ValueError('A variable name is required.')
            if value is not None and dimensions is None:
                msg = 'Variables with a value require dimensions. If this is a scalar variable, provide an empty list or ' \
                      'tuple.'
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

    def __setitem__(self, slc, variable):
        # tdk: order
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

    def __len__(self):
        return self.shape[0]

    @property
    def bounds(self):
        if self._bounds_name is None:
            ret = None
        else:
            ret = self.parent[self._bounds_name]
        return ret

    def set_bounds(self, value, force=False):
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
        return get_units_object(self.units)

    @property
    def dtype(self):
        if self._dtype is None:
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
        return self._get_dimensions_()

    @property
    def dimensions_dict(self):
        ret = OrderedDict()
        for d in self.dimensions:
            ret[d.name] = d
        return ret

    @property
    def dimension_names(self):
        return self._dimensions

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
        return self._get_fill_value_()

    def _get_fill_value_(self):
        return self._fill_value

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
        if self._mask is None:
            ret = False
        else:
            ret = True
        return ret

    @property
    def has_masked_values(self):
        """
        Return ``True`` if any values are masked.

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
        return self._parent is None

    @property
    def ndim(self):
        if self._dimensions is None:
            ret = 0
        else:
            ret = len(self._dimensions)
        return ret

    @property
    def resolution(self):
        # tdk: test
        # tdk: not sure where this belongs exactly. maybe on a value dimension?

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
        return self._get_shape_()

    @property
    def size(self):
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

    # tdk: remove me
    @property
    def masked_value(self):
        raise NotImplementedError

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
                    ret = variable_get_zeros(dimensions, self.dtype)
        return ret

    def set_value(self, value, update_mask=False):
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
        self._dtype = None
        self._fill_value = None

        if self.has_bounds:
            self.bounds.cfunits_conform(to_units, from_units=from_units)

    def convert_to_empty(self):
        if self.is_orphaned:
            self.set_mask(None)
            self.set_value(None)
            self._is_empty = True
        else:
            self.parent.convert_to_empty()

    def get_masked_value(self):
        if isinstance(self.dtype, ObjectType):
            dtype = object
        else:
            dtype = self.dtype
        ret = np.ma.array(self.get_value(), mask=self.get_mask(), dtype=dtype, fill_value=self.fill_value)
        return ret

    def reshape(self, *args):
        assert not self.has_bounds

        new_shape = [len(dimension) for dimension in args[0]]

        original_value = self.get_value()
        if self.has_mask:
            original_mask = self.get_mask()
        else:
            original_mask = None

        self.set_mask(None)
        self.set_value(None)
        self.set_dimensions(None)

        if original_mask is not None:
            new_mask = original_mask.reshape(*new_shape)
        else:
            new_mask = None

        self.set_dimensions(args[0])
        self.set_value(original_value.reshape(*new_shape))
        self.set_mask(new_mask)

    def set_extrapolated_bounds(self, name_variable, name_dimension):
        """Set the bounds variable using extrapolation."""

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
            # tdk: consider renaming this functions to get_bounds_from_2d
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
        return self._mask is not None

    @property
    def has_allocated_value(self):
        return self._value is not None

    @property
    def has_bounds(self):
        if not self.is_orphaned and self.bounds is not None:
            ret = True
        else:
            ret = False
        return ret

    def get_mask(self, create=False, check_value=False, eager=True):
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

    def allocate_value(self):
        if not self.is_empty:
            self.set_value(variable_get_zeros(self.dimensions, self.dtype))

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
        
        :param bool keep_bounds: If ``True``, maintain any bounds associated with the target variable. 
        :param bool clean_break: If ``True``, remove the target from the containing collection entirely. 
        :return: 
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
        # tdk: refactor to function
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

    def get_scatter_slices(self, splits):
        slices = create_nd_slices(splits, self.shape)
        return slices

    def get_value(self):
        if self._value is None:
            self._value = self._get_value_()
        return self._value

    def get_iter(self, **kwargs):
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
        implementations is in :class:`ocgis.new_interface.variable.SourcedVariable`
        """

    def _get_iter_value_(self):
        return self.get_value()

    def write(self, *args, **kwargs):
        """
        Write the field object to an open netCDF dataset object.

        :param dataset: The open dataset object or path for the write.
        :type dataset: :class:`netCDF4.Dataset` or str
        :param bool file_only: If ``True``, we are not filling the value variables. Only the file schema and dimension
         values will be written.
        :param bool unlimited_to_fixedsize: If ``True``, convert the unlimited dimension to fixed size.
        :param kwargs: Extra keyword arguments in addition to ``dimensions`` and ``fill_value`` to pass to
         ``createVariable``. See http://unidata.github.io/netcdf4-python/netCDF4.Dataset-class.html#createVariable
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


# tdk: variable collection should inherit from abstract container
class VariableCollection(AbstractNamedObject, AbstractCollection, Attributes):
    def __init__(self, name=None, variables=None, attrs=None, parent=None, children=None, aliases=None,
                 tags=None, source_name=constants.UNINITIALIZED, uid=None, is_empty=None):
        self._is_empty = is_empty
        self._dimensions = OrderedDict()
        self.parent = parent
        self.children = children or OrderedDict()

        if tags is None:
            tags = OrderedDict()
        self._tags = tags

        AbstractCollection.__init__(self)
        Attributes.__init__(self, attrs)
        AbstractNamedObject.__init__(self, name, aliases=aliases, source_name=source_name, uid=uid)

        if variables is not None:
            for variable in get_iter(variables, dtype=Variable):
                self.add_variable(variable)

    def __getitem__(self, item_or_slc):
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

    # tdk: dimensions and group can be removed with inheritance from abstractcontainer
    @property
    def dimensions(self):
        return self._dimensions

    @property
    def group(self):
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
    def is_empty(self):
        return get_is_empty_recursive(self)

    @property
    def shapes(self):
        return OrderedDict(
            [[k, v.shape] for k, v in list(self.items()) if not isinstance(v, CoordinateReferenceSystem)])

    # tdk: remove me
    @property
    def tags(self):
        raise NotImplementedError

    def add_child(self, child, force=False):
        if child.name in self.children and not force:
            raise ValueError("Child with name '{}' already in parent with name '{}'.".format(child.name, self.name))
        child.parent = self
        self.children[child.name] = child

    def add_dimension(self, dimension, force=False):
        existing_dim = self.dimensions.get(dimension.name)
        if existing_dim is not None and not force:
            if existing_dim != dimension:
                raise DimensionMismatchError(dimension.name, self.name)
        else:
            self.dimensions[dimension.name] = dimension

    def add_variable(self, variable, force=False):
        """
        :param :class:`ocgis.interface.base.variable.Variable`
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
        for v in list(self.values()):
            with orphaned(v):
                v.convert_to_empty()

    def copy(self):
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
        if tag in self._tags:
            raise ValueError('Tag "{}" already exists.'.format(tag))
        else:
            self._tags[tag] = []

    def get_by_tag(self, tag, create=False, strict=False):
        """
        :param str tag: The tag to retrieve.
        :param create bool: If ``True``, create the tag if it does not exist.
        :param strict bool: If ``True``, raise exception if variable name is not found in collection.
        :return: Tuple of variable objects that have the ``tag``.
        :rtype: tuple
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
        """Load all variable values from source."""

        for v in list(self.values()):
            v.load()

    def iter(self, **kwargs):
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
        names = get_dimension_names(dimensions)
        for var in list(self.values()):
            if len(set(var.dimension_names).intersection(names)) == len(names):
                yield var

    @staticmethod
    def read(*args, **kwargs):
        from ocgis import RequestDataset
        rd = RequestDataset(*args, **kwargs)
        return rd.driver.get_variable_collection()

    def remove_variable(self, variable):
        variable_name = get_variable_names(variable)[0]
        for v in list(self._tags.values()):
            if variable_name in v:
                v.remove(variable_name)
        self.pop(variable_name)

    def set_mask(self, variable, **kwargs):
        exclude = kwargs.pop(KeywordArgument.EXCLUDE, None)
        if exclude is not None:
            exclude = get_variable_names(exclude)
        update = kwargs.pop(KeywordArgument.UPDATE, False)
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
        self._storage = OrderedDict()
        self._dimensions = OrderedDict()
        self.children = OrderedDict()

    def write(self, *args, **kwargs):
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


def variable_get_zeros(dimensions, dtype):
    new_shape = get_dimension_lengths(dimensions)
    ret = np.zeros(new_shape, dtype=dtype)
    return ret
