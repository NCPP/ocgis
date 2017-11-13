import abc
from abc import abstractmethod

import numpy as np
import six
from pyproj import Proj, transform
from shapely.geometry import box

from ocgis import Variable, SourcedVariable, vm
from ocgis.base import raise_if_empty, is_field, AbstractInterfaceObject
from ocgis.constants import KeywordArgument, VariableName, WrapAction, DMK
from ocgis.exc import GridDeficientError
from ocgis.variable import crs
from ocgis.variable.base import AbstractContainer


class AbstractSpatialObject(AbstractInterfaceObject):
    """
    Superclass for spatial data objects having a coordinate system and a wrapped state.

    :keyword crs: (``=None``) Coordinate reference system for the spatial object.
    :type crs: :class:`ocgis.variable.crs.AbstractCRS`
    :keyword wrapped_state: (``='auto'``) Wrapped state of the object. If ``'auto'``, detect the wrapped state from the
     underlying data.
    :type wrapped_state: str | :class:`ocgis.constants.WrappedState`
    """

    def __init__(self, *args, **kwargs):
        if not is_field(self._get_field_()):
            raise TypeError('Host collection must be a field.')

        kwargs = kwargs.copy()
        self.crs = kwargs.pop(KeywordArgument.CRS, 'auto')

        self._wrapped_state = None
        self.wrapped_state = kwargs.pop(KeywordArgument.WRAPPED_STATE, 'auto')

        super(AbstractInterfaceObject, self).__init__(*args, **kwargs)

    @property
    def crs(self):
        return self.dimension_map.get_crs(parent=self._get_field_(), nullable=True)

    @crs.setter
    def crs(self, value):
        if value == 'auto':
            pass
        else:
            curr_crs = self.crs
            field = self._get_field_()
            if curr_crs is not None:
                field.pop(curr_crs.name)
            if value is not None:
                field.add_variable(value, force=True)
                value.format_spatial_object(self)
            self.dimension_map.set_crs(value)

    @property
    def dimension_map(self):
        field = self._get_field_()
        return field.dimension_map

    @dimension_map.setter
    def dimension_map(self, value):
        if value != 'auto':
            self._get_field_().dimension_map = value

    @property
    def wrapped_state(self):
        raise_if_empty(self)

        if self._wrapped_state == 'auto':
            if self.crs is None:
                ret = None
            else:
                ret = self.crs.get_wrapped_state(self)
        else:
            ret = self._wrapped_state
        return ret

    @wrapped_state.setter
    def wrapped_state(self, value):
        self._wrapped_state = value

    def unwrap(self):
        self._wrap_or_unwrap_(WrapAction.UNWRAP)

    def wrap(self):
        self._wrap_or_unwrap_(WrapAction.WRAP)

    def _get_field_(self):
        return self.parent

    @staticmethod
    def _get_parent_class_():
        from ocgis import Field
        return Field

    def _wrap_or_unwrap_(self, action):
        raise_if_empty(self)
        if self.crs is None or not self.crs.is_wrappable:
            raise TypeError("Coordinate system may not be wrapped/unwrapped.")
        else:
            self.crs.wrap_or_unwrap(action, self)


@six.add_metaclass(abc.ABCMeta)
class AbstractOperationsSpatialObject(AbstractSpatialObject):
    """
    Superclass for spatial objects exposing operations such as intersects, etc.
    """

    @property
    def envelope(self):
        """
        Create a Shapely polygon object from the object's extent.

        :rtype: :class:`shapely.geometry.polygon.Polygon`
        """
        return box(*self.extent)

    @property
    def extent(self):
        """
        Get the spatial extent of the object as a four element tuple ``(minx, miny, maxx, maxy)``.

        :rtype: tuple
        """
        return self._get_extent_()

    @property
    def extent_global(self):
        """
        Return the global extent of the grid collective across the current :class:`~ocgis.OcgVM`. The returned tuple is
        in the form: ``(minx, miny, maxx, maxy)``.

        :rtype: tuple
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """
        return get_extent_global(self)

    @abstractmethod
    def update_crs(self, to_crs):
        """
        Update the coordinate system in place.

        :param to_crs: The destination coordinate system.
        :type to_crs: :class:`~ocgis.variable.crs.AbstractCRS`
        """

        raise_if_empty(self)

        if self.crs is None:
            msg = 'The current CRS is None and cannot be updated. Has the coordinate system been assigned ' \
                  'appropriately?'
            raise ValueError(msg)
        if to_crs is None:
            msg = 'The destination CRS may not be None. Has the coordinate system been assigned appropriately?'
            raise ValueError(msg)

    def get_intersects(self, subset_geom, **kwargs):
        """
        Perform a spatial intersects operation on the grid and return a shallow copy.

        :param subset_geom: The subset Shapely geometry. All geometry types are accepted.
        :type subset_geom: :class:`shapely.geometry.base.BaseGeometry`
        :param kwargs: See :meth:`~ocgis.AbstractGrid.get_spatial_subset_operation`
        :rtype: :class:`~ocgis.AbstractGrid`
        """

        args = ['intersects', subset_geom]
        return self.get_spatial_subset_operation(*args, **kwargs)

    def get_intersection(self, subset_geom, **kwargs):
        """
        Perform a spatial intersection operation on the grid. A geometry variable is returned. An intersection
        operation modifies the grid underlying structure and regularity may no longer be guaranteed.

        :param subset_geom: The subset Shapely geometry. All geometry types are accepted.
        :type subset_geom: :class:`shapely.geometry.base.BaseGeometry`
        :param kwargs: See :meth:`~ocgis.AbstractGrid.get_spatial_subset_operation`
        :rtype: :class:`~ocgis.GeometryVariable`
        """

        args = ['intersection', subset_geom]
        return self.get_spatial_subset_operation(*args, **kwargs)

    @abstractmethod
    def get_nearest(self, target, return_indices=False):
        """Get nearest element to target geometry."""

    @abstractmethod
    def get_spatial_index(self):
        """Get the spatial index."""

    @abstractmethod
    def get_spatial_subset_operation(self, spatial_op, subset_geom, **kwargs):
        """Perform a spatial subset operation (this includes an intersection/clip)."""

    @abstractmethod
    def iter_records(self, use_mask=True):
        """Generate fiona-compatible records."""

    @abstractmethod
    def _get_extent_(self):
        """
        :returns: A tuple with order (minx, miny, maxx, maxy).
        :rtype: tuple
        """


@six.add_metaclass(abc.ABCMeta)
class AbstractSpatialContainer(AbstractContainer, AbstractOperationsSpatialObject):
    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        crs = kwargs.pop(KeywordArgument.CRS, 'auto')
        parent = kwargs.pop(KeywordArgument.PARENT, None)
        name = kwargs.pop(KeywordArgument.NAME, None)
        driver = kwargs.pop(KeywordArgument.DRIVER, None)
        assert len(kwargs) == 0

        if parent is not None and not is_field(parent):
            raise ValueError("'parent' object must be a field")

        AbstractContainer.__init__(self, name, parent=parent)
        AbstractOperationsSpatialObject.__init__(self, crs=crs)

        if driver is not None:
            self.parent.set_driver(driver)

    @property
    def dimension_map(self):
        return self.parent.dimension_map

    def get_mask(self, *args, **kwargs):
        """
        The grid's mask is stored independently from the coordinate variables' masks. The mask is actually a variable
        containing a mask. This approach ensures the mask may be persisted to file and retrieved/modified leaving all
        coordinate variables intact.

        .. note:: See :meth:`~ocgis.Variable.get_mask` for documentation.
        """
        from ocgis.spatial.grid import create_grid_mask_variable

        create = kwargs.get(KeywordArgument.CREATE, False)
        mask_variable = self.mask_variable
        ret = None
        if mask_variable is None:
            if create:
                mask_variable = create_grid_mask_variable(VariableName.SPATIAL_MASK, None, self.dimensions)
                self.set_mask(mask_variable)
        if mask_variable is not None:
            ret = mask_variable.get_mask(*args, **kwargs)
            if mask_variable.attrs.get('ocgis_role') != 'spatial_mask':
                msg = 'Mask variable "{}" must have an "ocgis_role" attribute with a value of "spatial_mask".'.format(
                    ret.name)
                raise ValueError(msg)
        return ret

    def set_mask(self, value, cascade=False):
        """
        Set the grid's mask from boolean array or variable.

        :param value: A mask array having the same shape as the grid. This may also be a variable with the same
         dimensions.
        :type value: :class:`numpy.ndarray` | :class:`~ocgis.Variable`
        :param cascade: If ``True``, cascade the mask along shared dimension on the grid.
        """
        from ocgis.spatial.grid import create_grid_mask_variable, grid_set_mask_cascade

        if isinstance(value, Variable):
            self.parent.add_variable(value, force=True)
            mask_variable = value
        else:
            mask_variable = self.mask_variable
            if mask_variable is None:
                mask_variable = create_grid_mask_variable(VariableName.SPATIAL_MASK, value, self.dimensions)
                self.parent.add_variable(mask_variable)
            else:
                mask_variable.set_mask(value)
        self.dimension_map.set_spatial_mask(mask_variable)

        if cascade:
            grid_set_mask_cascade(self)

    @staticmethod
    def _get_parent_class_():
        from ocgis import Field
        return Field


@six.add_metaclass(abc.ABCMeta)
class AbstractXYZSpatialContainer(AbstractSpatialContainer):
    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        x = kwargs.pop(KeywordArgument.X)
        y = kwargs.pop(KeywordArgument.Y)
        z = kwargs.pop(KeywordArgument.Z, None)
        mask = kwargs.pop(KeywordArgument.MASK, None)
        pos = kwargs.pop(KeywordArgument.POS, (0, 1))

        parent = kwargs.get(KeywordArgument.PARENT, None)

        # --------------------------------------------------------------------------------------------------------------

        if x is None:
            if parent is None:
                raise ValueError('A "parent" is required if no coordinate variables are provided.')
            x, y, z = self._get_xyz_from_parent_(parent)

        if x is None or y is None:
            if parent is not None:
                raise GridDeficientError("'x' or 'y' coordinates are missing.")
            else:
                raise ValueError("'x' and 'y' coordinates are required without a parent.")

        if x.dimensions is None or y.dimensions is None or (z is not None and z.dimensions is None):
            raise ValueError('Coordinate variables must have dimensions.')

        # --------------------------------------------------------------------------------------------------------------

        new_variables = [x, y]
        if z is not None:
            new_variables.append(z)
        if parent is None:
            parent = self._get_parent_class_()(variables=new_variables, force=True)
            kwargs[KeywordArgument.PARENT] = parent
        else:
            for var in new_variables:
                parent.add_variable(var, force=True)

        self._set_xyz_on_dimension_map_(x, y, z, pos, parent=parent)

        super(AbstractXYZSpatialContainer, self).__init__(**kwargs)

        if mask is not None:
            if not isinstance(mask, Variable):
                from ocgis.spatial.grid import create_grid_mask_variable
                mask = create_grid_mask_variable(VariableName.SPATIAL_MASK, mask, self.dimensions)
            self.parent.add_variable(mask, force=True)
            self.dimension_map.set_spatial_mask(mask)

    @property
    def archetype(self):
        """
        :return: archetype coordinate variable
        :rtype: :class:`~ocgis.Variable`
        """
        return self.y

    @property
    def coordinate_variables(self):
        """
        See :meth:`~ocgis.collection.field.Field.coordinate_variables`
        """
        ret = [self.x, self.y]
        z = self.z
        if z is not None:
            ret.append(z)
        return tuple(ret)

    @property
    def has_mask(self):
        """
        :return: ``True`` if the geometry abstraction variable is allocated on the grid.
        :rtype: bool
        """
        return self.mask_variable is not None

    @property
    def has_z(self):
        """
        :return: ``True`` if the grid has a z-coordinate.
        :rtype: bool
        """
        return self.z is not None

    @property
    def mask_variable(self):
        """
        :return: The mask variable associated with the grid. This will be ``None`` if no mask is present.
        :rtype: :class:`~ocgis.Variable`
        """
        ret = self.dimension_map.get_spatial_mask()
        if ret is not None:
            ret = self.parent[ret]
        return ret

    @property
    def shape_global(self):
        """
        Get the global shape across the current :class:`~ocgis.OcgVM`.

        :rtype: :class:`tuple` of :class:`int`
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """

        raise_if_empty(self)

        maxd = [max(d.bounds_global) for d in self.dimensions]
        shapes = vm.gather(maxd)
        if vm.rank == 0:
            shape_global = tuple(np.max(shapes, axis=0))
        else:
            shape_global = None
        shape_global = vm.bcast(shape_global)

        return shape_global

    @property
    def x(self):
        """
        Get or set the x-coordinate variable for the grid.

        :rtype: :class:`~ocgis.Variable`
        """
        ret = self._create_dimension_map_property_(DMK.X)
        return ret

    @property
    def y(self):
        """
        Get or set the y-coordinate variable for the grid.

        :rtype: :class:`~ocgis.Variable`
        """
        return self._create_dimension_map_property_(DMK.Y)

    @property
    def z(self):
        """
        Get or set the z-coordinate variable for the grid.

        :rtype: :class:`~ocgis.Variable`
        """
        return self._create_dimension_map_property_(DMK.LEVEL, nullable=True)

    def get_member_variables(self, include_bounds=True):
        """
        A spatial container is composed of numerous member variables defining coordinates, bounds, masks, and
        geometries. This method returns those variables if present on the current container object.

        :param include_bounds: If ``True``, include any bounds variables associated with the grid members.
        :rtype: :class:`list` of :class:`~ocgis.Variable`
        """
        targets = [self.x, self.y, self.z, self.mask_variable]

        ret = []
        for target in targets:
            if target is not None:
                ret.append(target)
                if include_bounds and target.has_bounds:
                    ret.append(target.bounds)
        return ret

    def update_crs(self, to_crs):
        super(AbstractXYZSpatialContainer, self).update_crs(to_crs)

        if isinstance(self.crs, crs.Cartesian) or isinstance(to_crs, crs.Cartesian):
            if isinstance(to_crs, crs.Cartesian):
                inverse = False
            else:
                inverse = True
            self.crs.transform_grid(to_crs, self, inverse=inverse)
        elif isinstance(self.crs, crs.CFRotatedPole):
            self.crs.update_with_rotated_pole_transformation(self, inverse=False)
        elif isinstance(to_crs, crs.CFRotatedPole):
            to_crs.update_with_rotated_pole_transformation(self, inverse=True)
        else:
            src_proj4 = self.crs.proj4
            dst_proj4 = to_crs.proj4

            src_proj4 = Proj(src_proj4)
            dst_proj4 = Proj(dst_proj4)

            y = self.y
            x = self.x

            value_row = self.y.get_value().reshape(-1)
            value_col = self.x.get_value().reshape(-1)

            tvalue_col, tvalue_row = transform(src_proj4, dst_proj4, value_col, value_row)

            self.x.set_value(tvalue_col.reshape(self.shape))
            self.y.set_value(tvalue_row.reshape(self.shape))

            if self.has_bounds:
                corner_row = y.bounds.get_value().reshape(-1)
                corner_col = x.bounds.get_value().reshape(-1)
                tvalue_col, tvalue_row = transform(src_proj4, dst_proj4, corner_col, corner_row)
                y.bounds.set_value(tvalue_row.reshape(y.bounds.shape))
                x.bounds.set_value(tvalue_col.reshape(x.bounds.shape))

        self.crs = to_crs

        self.crs.format_spatial_object(self, is_transform=True)

    def _get_canonical_dimension_map_(self, field=None, create=False):
        if field is None:
            field = self.parent
        return field.dimension_map.get_topology(self.topology, create=create)

    def _create_dimension_map_property_(self, entry_key, nullable=False):
        dimension_map = self._get_canonical_dimension_map_()
        ret = dimension_map.get_variable(entry_key, parent=self.parent, nullable=nullable)
        return ret

    def _get_xyz_from_parent_(self, parent):
        dmap = self._get_canonical_dimension_map_(field=parent, create=False)
        x = dmap.get_variable(DMK.X, parent=parent)
        y = dmap.get_variable(DMK.Y, parent=parent)
        z = dmap.get_variable(DMK.LEVEL, parent=parent, nullable=True)
        return x, y, z

    def _set_xyz_on_dimension_map_(self, x, y, z, pos, parent=None):
        if x.ndim == 2:
            x_pos = pos[1]
            y_pos = pos[0]
        else:
            x_pos, y_pos = [None, None]

        if z is not None and x.ndim == 2:
            dimensionless = True
        else:
            dimensionless = False

        if parent is None:
            parent = self.parent
        dimension_map = self._get_canonical_dimension_map_(field=parent, create=True)
        dimension_map.set_variable(DMK.X, x, pos=x_pos)
        dimension_map.set_variable(DMK.Y, y, pos=y_pos)
        if z is not None:
            dimension_map.set_variable(DMK.LEVEL, z, dimensionless=dimensionless)


@six.add_metaclass(abc.ABCMeta)
class AbstractSpatialVariable(SourcedVariable, AbstractOperationsSpatialObject):
    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        crs = kwargs.pop(KeywordArgument.CRS, 'auto')
        wrapped_state = kwargs.pop(KeywordArgument.WRAPPED_STATE, 'auto')
        parent = kwargs.get(KeywordArgument.PARENT, None)
        if parent is None:
            parent = AbstractOperationsSpatialObject._get_parent_class_()()
        kwargs[KeywordArgument.PARENT] = parent

        SourcedVariable.__init__(self, **kwargs)
        AbstractOperationsSpatialObject.__init__(self, crs=crs, wrapped_state=wrapped_state)

    def deepcopy(self, eager=False):
        ret = super(AbstractSpatialVariable, self).deepcopy(eager)
        if ret.crs is not None:
            ret.crs = ret.crs.deepcopy()
        return ret

    def extract(self, **kwargs):
        crs = self.crs
        if crs is not None:
            crs = crs.copy()
        ret = super(AbstractSpatialVariable, self).extract(**kwargs)
        if crs is not None:
            ret.parent.add_variable(crs)
        return ret


def get_extent_global(container):
    raise_if_empty(container)

    extent = container.extent
    extents = vm.gather(extent)

    if vm.rank == 0:
        extents = [e for e in extents if e is not None]
        extents = np.array(extents)
        ret = [None] * 4
        ret[0] = np.min(extents[:, 0])
        ret[1] = np.min(extents[:, 1])
        ret[2] = np.max(extents[:, 2])
        ret[3] = np.max(extents[:, 3])
        ret = tuple(ret)
    else:
        ret = None
    ret = vm.bcast(ret)

    return ret
