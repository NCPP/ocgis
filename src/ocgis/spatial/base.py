import abc
import itertools
from abc import abstractmethod

import numpy as np
import six
from ocgis import Variable, SourcedVariable, vm
from ocgis.base import raise_if_empty, is_field, AbstractInterfaceObject
from ocgis.constants import KeywordArgument, VariableName, WrapAction, DMK
from ocgis.exc import GridDeficientError
from ocgis.variable import crs
from ocgis.variable.base import AbstractContainer
from pyproj import Proj, transform
from shapely.geometry import box


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

    def unwrap(self, force=False):
        self._wrap_or_unwrap_(WrapAction.UNWRAP, force=force)

    def wrap(self, force=False):
        self._wrap_or_unwrap_(WrapAction.WRAP, force=force)

    def _get_field_(self):
        return self.parent

    @staticmethod
    def _get_parent_class_():
        from ocgis import Field
        return Field

    def _wrap_or_unwrap_(self, action, force=False):
        raise_if_empty(self)
        if self.crs is None or not self.crs.is_wrappable:
            raise TypeError("Coordinate system may not be wrapped/unwrapped.")
        else:
            self.crs.wrap_or_unwrap(action, self, force=force)


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
    def update_crs(self, to_crs, from_crs=None):
        """
        Update the coordinate system in place.

        :param to_crs: The destination coordinate system.
        :type to_crs: :class:`~ocgis.variable.crs.AbstractCRS`
        :param from_crs: Optional original coordinate system to temporarily assign to the data. Useful when the
         object's coordinate system is different from the desired coordinate system.
        :type from_crs: :class:`~ocgis.variable.crs.AbstractCRS`
        """

        raise_if_empty(self)

        if self.crs is None and from_crs is None:
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
        A spatial container's mask is stored independently from the coordinate variables' masks. The mask is actually a
        variable containing a mask. This approach ensures the mask may be persisted to file and retrieved/modified
        leaving all coordinate variables intact.

        .. note:: See :meth:`~ocgis.Variable.get_mask` for documentation.
        """
        args = list(args)
        args.insert(0, self)
        return self.driver.get_or_create_spatial_mask(*args, **kwargs)

    def set_mask(self, value, cascade=False):
        """
        Set the spatial container's mask from a boolean array or variable.

        :param value: A mask array having the same shape as the grid. This may also be a variable with the same
         dimensions.
        :type value: :class:`numpy.ndarray` | :class:`~ocgis.Variable`
        :param cascade: If ``True``, cascade the mask along shared dimensions on the spatial container.
        """
        self.driver.set_spatial_mask(self, value, cascade=cascade)

    @staticmethod
    def _get_parent_class_():
        from ocgis import Field
        return Field


@six.add_metaclass(abc.ABCMeta)
class AbstractXYZSpatialContainer(AbstractSpatialContainer):
    """
    Abstract container for X, Y, and optionally Z coordinate variables. If ``x`` and ``y`` are not provided, then
    ``parent`` is required.

    :keyword x: (``=None``) X-coordinate variable
    :type x: :class:`ocgis.Variable`
    :keyword y: (``=None``) Y-coordinate variable
    :type y: :class:`ocgis.Variable`
    :keyword parent: (``=None``) Parent field object
    :type parent: :class:`ocgis.Field`
    :keyword mask: (``=None``) Mask variable
    :type mask: :class:`ocgis.Variable`
    :keyword pos: (``=(0, 1)``) Axis values for n-dimensional coordinate arrays
    :type pos: tuple
    :keyword is_isomorphic: See ``grid_is_isomorphic`` documentation for :class:`ocgis.Field`
    """

    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        x = kwargs.pop(KeywordArgument.X, None)
        y = kwargs.pop(KeywordArgument.Y, None)
        z = kwargs.pop(KeywordArgument.Z, None)
        mask = kwargs.pop(KeywordArgument.MASK, None)
        pos = kwargs.pop(KeywordArgument.POS, None)
        is_isomorphic = kwargs.pop(DMK.IS_ISOMORPHIC, 'auto')

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

        if pos is None:
            pos = parent.driver.default_axes_positions
        self._set_xyz_on_dimension_map_(x, y, z, pos, parent=parent)

        super(AbstractXYZSpatialContainer, self).__init__(**kwargs)

        if mask is not None:
            if not isinstance(mask, Variable):
                mask = create_spatial_mask_variable(VariableName.SPATIAL_MASK, mask, self.dimensions)
            self.parent.add_variable(mask, force=True)
            self.dimension_map.set_spatial_mask(mask)

        # XYZ containers are not considered isomorphic (repeated topology or shapes) by default.
        if is_isomorphic == 'auto':
            if self.dimension_map.get_property(DMK.IS_ISOMORPHIC) is None:
                is_isomorphic = False
        if is_isomorphic != 'auto':
            self.is_isomorphic = is_isomorphic

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
    def has_mask_global(self):
        """
        Returns ``True`` if the global spatial object has a mask. Collective across the current VM.

        :rtype: bool
        """
        raise_if_empty(self)
        has_masks = vm.gather(self.has_mask)
        if vm.rank == 0:
            has_mask = np.any(has_masks)
        else:
            has_mask = None
        has_mask = vm.bcast(has_mask)
        return has_mask

    @property
    def has_masked_values(self):
        """
        Returns ``True`` if the spatial object's mask contains any masked values. Will return ``False`` if the object
        has no mask.

        :rtype: bool
        """
        if self.has_mask:
            ret = self.get_mask().any()
        else:
            ret = False
        return ret

    @property
    def has_masked_values_global(self):
        """
        Returns ``True`` if the global spatial object's mask contains any masked values. Will return ``False`` if the
        global object has no mask. Collective across the current VM.

        :rtype: bool
        """
        raise_if_empty(self)
        has_masks = vm.gather(self.has_masked_values)
        if vm.rank == 0:
            ret = np.any(has_masks)
        else:
            ret = None
        ret = vm.bcast(ret)
        return ret

    @property
    def has_z(self):
        """
        :return: ``True`` if the grid has a z-coordinate.
        :rtype: bool
        """
        return self.z is not None

    @property
    def is_isomorphic(self):
        """See ``grid_is_isomorphic`` documentation for :class:`ocgis.Field`"""
        return self.dimension_map.get_property(DMK.IS_ISOMORPHIC)

    @is_isomorphic.setter
    def is_isomorphic(self, value):
        self.dimension_map.set_property(DMK.IS_ISOMORPHIC, value)

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
    def resolution(self):
        """
        Returns the average spatial along resolution along the ``x`` and ``y`` dimensions.

        :rtype: float
        """
        if self.is_isomorphic:
            if 1 in self.shape:
                if self.shape[0] != 1:
                    ret = self.resolution_y
                elif self.shape[1] != 1:
                    ret = self.resolution_x
                else:
                    raise NotImplementedError(self.shape)
            else:
                ret = np.mean([self.resolution_y, self.resolution_x])
        else:
            raise NotImplementedError('Resolution not defined when "self.is_isomorphic=False"')

        return ret

    @property
    def resolution_max(self):
        """
        Returns the maximum spatial resolution between the ``x`` and ``y`` coordinate variables.

        :rtype: float
        """
        return max([self.resolution_x, self.resolution_y])

    @property
    def resolution_x(self):
        """
        Returns the resolution ox ``x`` variable.

        :rtype: float
        """
        return self.driver.array_resolution(self.x.get_value(), 1)

    @property
    def resolution_y(self):
        """
        Returns the resolution ox ``y`` variable.

        :rtype: float
        """
        return self.driver.array_resolution(self.y.get_value(), 0)

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

    def update_crs(self, to_crs, from_crs=None):
        super(AbstractXYZSpatialContainer, self).update_crs(to_crs, from_crs=from_crs)

        if from_crs is None:
            from_crs = self.crs

        if isinstance(self.crs, crs.Cartesian) or isinstance(to_crs, crs.Cartesian):
            if isinstance(to_crs, crs.Cartesian):
                inverse = False
            else:
                inverse = True
            from_crs.transform_grid(to_crs, self, inverse=inverse)
        elif isinstance(self.crs, crs.CFRotatedPole):
            from_crs.update_with_rotated_pole_transformation(self, inverse=False)
        elif isinstance(to_crs, crs.CFRotatedPole):
            to_crs.update_with_rotated_pole_transformation(self, inverse=True)
        else:
            src_proj4 = from_crs.proj4
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

    def _create_dimension_map_property_(self, entry_key, nullable=False):
        dimension_map = self._get_canonical_dimension_map_()
        ret = dimension_map.get_variable(entry_key, parent=self.parent, nullable=nullable)
        return ret

    def _gc_iter_dst_grid_slices_(self, grid_chunker, yield_idx=None):
        return self.driver._gc_iter_dst_grid_slices_(grid_chunker, yield_idx=yield_idx)

    def _gc_nchunks_dst_(self, grid_chunker):
        return self.driver._gc_nchunks_dst_(grid_chunker)

    def _get_canonical_dimension_map_(self, field=None, create=False):
        if field is None:
            field = self.parent
        return field.dimension_map.get_topology(self.topology, create=create)

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
class AbstractSpatialVariable(AbstractOperationsSpatialObject, SourcedVariable):
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


def create_spatial_mask_variable(name, mask_value, dimensions):
    """
    Create an OCGIS spatial mask variable with standard attributes. By default, the value of the returned variable is
    allocated with zeroes.

    :param str name: Variable name
    :param mask_value: Boolean array with dimension matching ``dimensions``
    :type mask_value: :class:`numpy.ndarray`
    :param dimensions: Dimension sequence for the new variable
    :type dimensions: tuple(:class:`ocgis.Dimension`, ...)
    :rtype: :class:`ocgis.Variable`
    """
    mask_variable = Variable(name, mask=mask_value, dtype=np.dtype('i1'), dimensions=dimensions,
                             attrs={'ocgis_role': 'spatial_mask',
                                    'description': 'values matching fill value are spatially masked'})
    mask_variable.allocate_value(fill=0)
    return mask_variable


def create_split_polygons(geom, split_shape):
    minx, miny, maxx, maxy = geom.bounds
    rows = np.linspace(miny, maxy, split_shape[0] + 1)
    cols = np.linspace(minx, maxx, split_shape[1] + 1)

    return create_split_polygons_from_meshgrid_vectors(cols, rows)


def create_split_polygons_from_meshgrid_vectors(cols, rows):
    nrow = rows.size - 1
    ncol = cols.size - 1
    fill = np.zeros(nrow * ncol, dtype=object)
    for fillidx, (rowidx, colidx) in enumerate(itertools.product(range(nrow), range(ncol))):
        minx = cols[colidx]
        miny = rows[rowidx]
        maxx = cols[colidx + 1]
        maxy = rows[rowidx + 1]
        fill[fillidx] = box(minx, miny, maxx, maxy)

    return fill


def get_extent_global(container):
    raise_if_empty(container)

    extent = container.extent
    extents = vm.gather(extent)

    # ocgis_lh(msg='extents={}'.format(extents), logger='spatial.base', level=logging.DEBUG)

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


def iter_spatial_decomposition(sobj, splits, **kwargs):
    """
    Yield spatial subsets of the target ``sobj`` defined by the spatial decomposition created from ``splits``.

    This method is collective across the current :class:`ocgis.OcgVM`.

    :param sobj: Target XYZ spatial container to subset
    :type sobj: :class:`ocgis.spatial.base.AbstractXYZSpatialContainer`
    :param tuple splits: The number of splits along each dimension of the ``sobj``'s global spatial extent
    :param kwargs: See keyword arguments for :meth:`ocgis.spatial.base.AbstractXYZSpatialContainer.get_intersects`
    :rtype: :class:`ocgis.spatial.base.AbstractXYZSpatialContainer`
    """
    kwargs = kwargs.copy()
    yield_idx = kwargs.pop('yield_idx', None)
    kwargs[KeywordArgument.RETURN_SLICE] = True

    # Adjust the split definition to work with polygon creation call. --------------------------------------------------
    len_splits = len(splits)
    # Only splits along two dimensions.
    assert (len_splits <= 2)
    if len_splits == 1:
        split_shape = (splits[0], 1)
    else:
        split_shape = splits
    # ------------------------------------------------------------------------------------------------------------------

    # For each split polygon, subset the target spatial object and yield it. -------------------------------------------
    extent_global = sobj.extent_global
    bbox = box(*extent_global)
    split_polygons = create_split_polygons(bbox, split_shape)
    for ctr, sp in enumerate(split_polygons):
        if yield_idx is not None:
            if ctr < yield_idx:
                continue
            else:
                break
        yield sobj.get_intersects(sp, **kwargs)
    # ------------------------------------------------------------------------------------------------------------------
