import abc
import logging
from collections import deque

import numpy as np
import six
from ocgis import env, vm
from ocgis.base import raise_if_empty, is_unstructured_driver, get_dimension_names
from ocgis.constants import KeywordArgument, GridAbstraction, VariableName, AttributeName, GridChunkerConstants, \
    RegriddingRole, DMK, MPITag, DriverKey, ConversionTarget, MPI_EMPTY_VALUE
from ocgis.exc import RequestableFeature
from ocgis.spatial.base import AbstractXYZSpatialContainer
from ocgis.util.helpers import get_formatted_slice, arange_from_dimension, create_unique_global_array, is_xarray
from ocgis.util.logging_ocgis import ocgis_lh
from ocgis.variable.base import get_dslice, Variable
from ocgis.variable.dimension import create_distributed_dimension
from ocgis.variable.geom import GeometryProcessor, GeometryVariable
from ocgis.vmachine.mpi import cancel_free_requests
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry
from shapely.geometry.linestring import LineString
from shapely.geometry.multipolygon import MultiPolygon


def format_gridunstruct_return(func):
    """Allows operations on geometry coordinate objects to return their parent grid."""

    def wrapped(*args, **kwargs):
        obj = args[0]
        result = func(*args, **kwargs)
        if obj.hosted:
            if isinstance(result, tuple):
                to_format = result[0]
                is_tuple = True
            else:
                to_format = result
                is_tuple = False
            if isinstance(to_format, AbstractGeometryCoordinates):
                to_format = to_format.parent.grid
            if is_tuple:
                ret = tuple([to_format] + list(result)[1:])
            else:
                ret = to_format
        else:
            ret = result
        return ret

    return wrapped


@six.add_metaclass(abc.ABCMeta)
class AbstractGeometryCoordinates(AbstractXYZSpatialContainer):
    """
    Superclass for geometry coordinate objects. These objects manage coordinate arrays for subsetting and conversion.

    :param x: The x-coordinate variable. Required if no parent is provided.
    :type x: :class:`~ocgis.Variable`
    :param y: The y-coordinate variable. Required if no parent is provided.
    :type y: :class:`~ocgis.Variable`
    :param z: The z-coordinate variable. Not required.
    :type z: :class:`~ocgis.Variable`
    :param cindex: The element node connectivity variable. If provided, this is used to index into coordinate variables
     ``x``, ``y``, and/or ``z``. If this is ``None``, use coordinate variables' first dimension as the element
     dimension (ragged arrays). If ``'auto'`` (the default), attempt to retrieve an appropriate variable from the
     dimension map.
    :type cindex: None | :class:`~ocgis.Variable` | str
    :param bool packed: If ``True``, the element node connectivity variable has been de-duplicated.
    :param start_index: If ``'auto'``, attempt to retrieve this value from the element node connectivity variable. The
     default is ``0`` if it cannot be found. An integer may also be provided.
    :type start_index: int | str
    :param bool hosted: If ``False``, this object is not hosted by an unstructured grid. If ``True``, it is hosted by
     an unstructured grid object. Hosted objects will return their parents for some classes of operations.
    :param dict kwargs: Optional keyword arguments to the superclass.
    """

    def __init__(self, x=None, y=None, z=None, cindex='auto', packed=True, start_index='auto', hosted=False, **kwargs):
        self._start_index = start_index

        self.packed = packed
        self.hosted = hosted

        kwargs = kwargs.copy()
        kwargs[KeywordArgument.X] = x
        kwargs[KeywordArgument.Y] = y
        kwargs[KeywordArgument.Z] = z

        # The mask requires working with the element dimension which is dependent on the element connectivity index if
        # present.
        mask = kwargs.pop(KeywordArgument.MASK, None)

        super(AbstractGeometryCoordinates, self).__init__(**kwargs)

        # Always overload the driver to UGRID if the current driver is not unstructured.
        driver_klass = self.dimension_map.get_driver(as_class=True)
        if not is_unstructured_driver(driver_klass):
            self.dimension_map.set_driver(DriverKey.NETCDF_UGRID)

        if cindex == 'auto':
            dmap = self._get_canonical_dimension_map_(field=self.parent)
            cindex = dmap.get_variable(DMK.ELEMENT_NODE_CONNECTIVITY, parent=self.parent, nullable=True)
        self.cindex = cindex

        if cindex is not None:
            name_ed = get_dimension_names(self.element_dim)[0]
            name_nd = get_dimension_names(self.node_dim)[0]
            if not self.is_empty and name_ed == name_nd and self.abstraction != GridAbstraction.POINT:
                msg = 'The element and node dimensions must have different names.'
                raise ValueError(msg)

        # Set the spatial mask following work with element connectivity to avoid a race condition.
        if mask is not None:
            self.set_mask(mask)

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, self.ndim)
        if not isinstance(slc, dict):
            slc = get_dslice(self.dimensions, slc)
        ret = self.copy()
        new_parent = ret.parent[slc]
        ret.parent = new_parent
        return ret

    @abc.abstractproperty
    def abstraction(self):
        """
        Return the generic abstraction for the geometry coordinates following :class:`~ocgis.constants.GridAbstraction.`
        """

    @property
    def cindex(self):
        """
        Provides an index into coordinate variables to extract coordinate values for elements. When setting, the first
        dimension is considered the representative element count dimension.

        :rtype: None | :class:`~ocgis.Variable`
        """

        dmap = self._get_canonical_dimension_map_(self.parent, create=False)
        return dmap.get_variable(DMK.ELEMENT_NODE_CONNECTIVITY, parent=self.parent, nullable=True)

    @cindex.setter
    def cindex(self, value):
        """
        :param value: The coordinate index variable of integer type.
        :type value: `~ocgis.Variable` || NoneType
        """

        dmap = self._get_canonical_dimension_map_(self.parent, create=False)
        dmap.set_variable(DMK.ELEMENT_NODE_CONNECTIVITY, value, pos=0)
        if value is not None:
            self.parent.add_variable(value, force=True)
            if AttributeName.START_INDEX not in value.attrs:
                if self._start_index == 'auto':
                    sindex = self.driver._start_index
                else:
                    sindex = self._start_index
                value.attrs[AttributeName.START_INDEX] = sindex

    @property
    def element_dim(self):
        """
        Get the element dimension. The size of the dimension is equivalent to the element count.

        :rtype: :class:`~ocgis.Dimension`
        """

        raise_if_empty(self)
        return self.parent.driver.get_element_dimension(self)

    @property
    def has_bounds(self):
        """
        Always return ``False``. Geometry coordinate objects will never have bounds.

        :rtype: bool
        """
        return False

    @property
    def has_multi(self):
        """
        If ``True``, this object represents multi-geometries (multi-polygon for example).

        :rtype: bool
        """
        return self.multi_break_value is not None

    @property
    def is_vectorized(self):
        """
        Always return ``False``. Geometry coordinates are never vectorized.

        :rtype: bool
        """
        return False

    @property
    def multi_break_value(self):
        """
        The break value to use for determining multi-geometries.

        :return: int | None
        """
        return self.parent.driver.get_multi_break_value(self.cindex)

    @property
    def ndim(self):
        """
        Get the representative dimension. This is typically ``1``.

        :rtype: int
        """
        return self.archetype.ndim

    @property
    def node_dim(self):
        """
        Get the node dimension.

        :rtype: :class:`~ocgis.Dimension`
        """

        return self.archetype.dims[0]

    @property
    def shape(self):
        """
        Get the current size of the element dimension as a tuple. For example: ``(5,)``.

        :rtype: tuple
        """

        return tuple([self.element_dim.size_current])

    @property
    def start_index(self):
        """
        Get the start index. ``0`` is the default. For data written and read from Fortran natively, this is often ``1``.
        This will attempt to read an attribute from the element node connectivity variable called
        :attr:`ocgis.constants.AttributeName.START_INDEX`.

        :rtype: int
        """
        if self._start_index == 'auto':
            cindex = self.cindex
            if cindex is not None:
                ret = cindex.attrs[AttributeName.START_INDEX]
            else:
                ret = 0
        else:
            ret = self._start_index
        return ret

    @property
    def topology(self):
        """Alias of :attr:`~ocgis.spatial.geomc.AbstractGeometryCoordinates.abstraction`."""
        return self.abstraction

    @abc.abstractproperty
    def __shapely_geometry_class__(self):
        """
        Return the class to use for constructing Shapely geometry objects.

        >>> return Polygon

        :return: :class:`~shapely.geometry.base.BaseGeometry`
        """

    @property
    def __shapely_multipart_class__(self):
        """
        Return the Shapely multipart geometry class to use for creating multi-geometry objects.

        >>> return MultiPolygon

        :return: :class:`~shapely.geometry.base.BaseMultipartGeometry`
        """
        raise NotImplementedError

    @abc.abstractproperty
    def __use_bounds_intersects_optimizations__(self):
        """Return ``True`` if bounds optimizations should be used."""

    def convert_to(self, target=ConversionTarget.GEOMETRY_VARIABLE, **kwargs):
        """
        Convert the geometry coordinate object to various targets.

        :param target: The destination conversion target.
        :type target: :attr:`~ocgis.constants.ConversionTarget`
        :param dict kwargs: Keyword arguments for the creation of the destination object.
        :return: Varies depending on the conversion target.
        """
        kwargs = kwargs.copy()
        if target == ConversionTarget.GEOMETRY_VARIABLE:
            from ocgis.variable.geom import GeometryVariable
            kwargs[KeywordArgument.VALUE] = list(self.iter_geometries(with_index=False))
            kwargs[KeywordArgument.DIMENSIONS] = [self.element_dim]
            if self.crs is not None:
                kwargs[KeywordArgument.CRS] = self.crs
            ret = GeometryVariable(**kwargs)
        else:
            raise RequestableFeature(target)
        return ret

    def get_geometry_iterable(self, use_mask=True, hint_mask=None, use_memory_optimizations=None, with_index=True):
        """
        Yield a tuple composed of the current iterator index and Shapely geometry object. If the geometry is masked,
        the geometry will be ``None``. For example: ``(2, <Polygon>)`` or ``(3, None)`` if masked.

        :param bool use_mask: If ``True``, use a mask for iteration. This is retrieved from the object if ``hint_mask``
         is ``None``. If ``False``, do not use the mask for iteration yielding underlying data if the object is mask.
        :param hint_mask: If ``None``, use the object's mask. If a boolean array, use this as the mask as opposed to
         the object's mask. The array must have the same dimension as ``self``.
        :type hint_mask: None | :class:`numpy.ndarray`
        :param use_memory_optimizations: If ``None``, default to :attr:`ocgis.env.USE_MEMORY_OPTIMIZATIONS`.
         If ``True``, do not eagerly load coordinates. If ``False``, load all coordinates into memory improving
         performance by limiting IO.
        :type use_memory_optimizations: None | bool
        :param bool with_index: If ``False``, do not yield the current iteration index.
        :return: tuple(int, <Shapely geometry>) | <Shapely geometry>
        """
        if use_memory_optimizations is None:
            use_memory_optimizations = env.USE_MEMORY_OPTIMIZATIONS
        if use_memory_optimizations and self.cindex is not None:
            use_memory_optimizations = True
        else:
            use_memory_optimizations = False

        if self.cindex is None:
            cindex = None
            has_multi = False
            mbv = None
        else:
            cindex = self.cindex.get_value()
            has_multi = self.has_multi
            mbv = self.multi_break_value
            if mbv is not None:
                assert mbv < 0

        xvar = self.x.extract()
        yvar = self.y.extract()
        if self.has_z:
            zvar = self.z.extract()

        if not use_memory_optimizations:
            x_value = xvar.get_value()
            y_value = yvar.get_value()

        if self.has_z and not use_memory_optimizations:
            z_value = zvar.get_value()
        else:
            z_value = None

        if self.has_mask:
            mask_value = self.get_mask()
        else:
            mask_value = None
        if hint_mask is not None:
            if mask_value is None:
                mask_value = hint_mask
            else:
                mask_value = np.logical_or(mask_value, hint_mask)

        has_mask = self.has_mask
        has_z = self.has_z
        get_shapely_geometry = self.get_shapely_geometry
        get_element_node_connectivity_by_index = self.get_element_node_connectivity_by_index
        start_index = self.start_index
        for idx in range(len(self.element_dim)):
            if use_mask and has_mask:
                is_masked = mask_value[idx]
            else:
                is_masked = False
            geom = None
            if not is_masked:
                if cindex is not None:
                    ec_idx = get_element_node_connectivity_by_index(cindex, idx)
                    if start_index > 0:
                        ec_idx -= start_index
                else:
                    ec_idx = idx

                collected = deque()
                if has_multi:
                    mitr = iter_multipart_coordinates(ec_idx, mbv)
                else:
                    mitr = [ec_idx]

                for comp_coords in mitr:
                    z = None
                    if use_memory_optimizations:
                        x = xvar[comp_coords].get_value()[0]
                        y = yvar[comp_coords].get_value()[0]
                        if has_z:
                            z = zvar[comp_coords].get_value()[0]
                    else:
                        x = x_value[comp_coords]
                        y = y_value[comp_coords]
                        if has_z:
                            z = z_value[comp_coords]
                    if has_z:
                        geom = get_shapely_geometry(x, y, z)
                    else:
                        geom = get_shapely_geometry(x, y)
                    collected.append(geom)

                if has_multi:
                    geom = self.__shapely_multipart_class__(collected)
                else:
                    geom = collected[0]

            if with_index:
                yield idx, geom
            else:
                yield geom

    def get_distributed_slice(self, slc, **kwargs):
        """
        Slice a distributed object.

        :param slc: A slice-like object.
        :type slc: <varying>
        :param dict kwargs: Optional arguments to :meth:`~ocgis.Variable.get_distributed_slice`.
        :rtype: :class:`ocgis.spatial.geomc.AbstractGeometryCoordinates`
        """

        slc = get_formatted_slice(slc, self.ndim)
        if self.cindex is None:
            target = self.x
        else:
            target = self.cindex
        if len(slc) != target.ndim:
            dslc = get_dslice(self.dimensions, slc)
            for dim in target.dimensions:
                if dim.name not in dslc:
                    dslc[dim.name] = slice(None)
            slc = [dslc[dim.name] for dim in target.dimensions]
        new_parent = target.get_distributed_slice(slc, **kwargs).parent
        ret = self.copy()
        ret.parent = new_parent
        return ret

    @abc.abstractmethod
    def get_element_node_connectivity_by_index(self, element_connectivity, idx):
        """
        Return something that can be used for indexing into coordinate arrays to retrieve the coordinates for the
        current element.

        :param element_connectivity: An element connectivity array with the first dimension/axis as the element
         dimension.
        :type element_connectivity: :class:`numpy.ndarray`
        :param int idx: The element index.
        :rtype: <used as NumPy index>
        """

    def get_nearest(self, target, return_indices=False):
        raise RequestableFeature("'get_nearest' is not implemented.")

    def get_shapely_geometry(self, *args, **kwargs):
        """
        Return a Shapely geometry object.

        :rtype: :class:`shapely.geometry.base.BaseGeometry`
        """

        return self.__shapely_geometry_class__(*args, **kwargs)

    def get_spatial_index(self):
        raise NotImplementedError

    @format_gridunstruct_return
    def get_spatial_subset_operation(self, spatial_op, subset_geom, return_slice=False, original_mask=None,
                                     keep_touches=True, cascade=True, optimized_bbox_subset=False, apply_slice=True,
                                     geom_name=None):
        """
        Perform intersects or intersection operations on the object.

        :param str spatial_op: Either an ``'intersects'`` or an ``'intersection'`` spatial operation.
        :param subset_geom: A scalar (single geometry) geometry variable or Shapely geometry to use in the spatial
         operation. All geometry types are accepted.
        :type subset_geom: :class:`~ocgis.GeometryVariable` | :class:`shapely.geometry.base.BaseGeometry`
        :param bool return_slice: If ``True``, also return the slices used to limit the grid's extent.
        :param original_mask: An optional mask to use as a hint for spatial operation. ``True`` values are excluded
         from spatial consideration.
        :type original_mask: :class:`numpy.ndarray`
        :param keep_touches: If ``True`` (the default), keep geometries that touch the subset geometry.
        :type keep_touches: :class:`bool`
        :param cascade: If ``True`` (the default), set the mask across all variables in the grid's parent collection.
        :param optimized_bbox_subset: If ``True``, perform an optimized bounding box subset on the grid. This will only
         use the grid's representative coordinates ignoring bounds, geometries, etc.
        :param bool apply_slice: If ``True`` (the default), apply the slice to the grid object in addition to updating
         its mask.
        :param str geom_name: If provided, use this name for the output geometry variable if this is an intersection
         operation.
        :return: If ``return_slice`` is ``False`` (the default), return a shallow copy of the sliced grid. If
         ``return_slice`` is ``True``, this will be a tuple with the subsetted object as the first element and the slice
         used as the second. If ``spatial_op`` is ``'intersection'``, the returned object is a geometry variable.
        :rtype: :class:`~ocgis.Grid` | :class:`~ocgis.GeometryVariable` | :class:`tuple` of ``(<returned object>, <slice used>)``
        """
        # TODO: Merge this with the grid's spatial operation.
        if optimized_bbox_subset and spatial_op == 'intersection':
            raise ValueError("'optimized_bbox_subset' must be False when performing an intersection")

        raise_if_empty(self)

        try:
            subset_geom = subset_geom.prepare()
        except AttributeError:
            if not isinstance(subset_geom, BaseGeometry):
                msg = 'Only Shapely geometries allowed for subsetting. Subset type is "{}".'.format(
                    type(subset_geom))
                raise ValueError(msg)
        else:
            subset_geom = subset_geom.get_value()[0]

        if self.get_mask() is None:
            original_has_mask = False
        else:
            original_has_mask = True

        if geom_name is None:
            geom_name = get_default_geometry_variable_name(self)

        if spatial_op == 'intersection':
            perform_intersection = True
        else:
            perform_intersection = False

        if original_mask is None and self.__use_bounds_intersects_optimizations__:
            if isinstance(subset_geom, BaseMultipartGeometry):
                geom_itr = subset_geom
            else:
                geom_itr = [subset_geom]

            x = self.x.get_value()
            y = self.y.get_value()
            if self.has_z:
                z = self.z.get_value()
            else:
                z = None
            for ctr, geom in enumerate(geom_itr):
                if geom.has_z:
                    coords = np.array(geom.exterior.coords)
                    z_coords = coords[:, 2]
                    z_bounds = z_coords.min(), z_coords.max()
                else:
                    z_bounds = None
                single_hint_mask = get_xyz_select(x, y, geom.bounds, z=z, z_bounds=z_bounds, keep_touches=keep_touches)

                if ctr == 0:
                    hint_mask = single_hint_mask
                else:
                    hint_mask = np.logical_or(hint_mask, single_hint_mask)

            hint_mask = np.invert(hint_mask)

            original_mask = hint_mask
            if not optimized_bbox_subset:
                mask = self.get_mask()
                if mask is not None:
                    original_mask = np.logical_or(mask, hint_mask)
        elif not self.__use_bounds_intersects_optimizations__:
            original_mask = np.zeros(self.element_dim.size, dtype=bool)

        ret = self.copy()
        if original_has_mask:
            ret.set_mask(ret.get_mask().copy())

        if optimized_bbox_subset:
            the_slice = np.invert(original_mask)
            sliced_obj = ret.get_distributed_slice(the_slice)
        else:
            fill_mask = original_mask
            geometry_fill = None
            # If everything is masked, there is no reason to load the grid geometries.
            if not original_mask.all():
                if perform_intersection:
                    geometry_fill = np.zeros(fill_mask.shape, dtype=object)
                gp = GeometryProcessor(self.get_geometry_iterable(hint_mask=original_mask), subset_geom,
                                       keep_touches=keep_touches)
                for idx, intersects_logical, current_geometry in gp.iter_intersects():
                    fill_mask[idx] = not intersects_logical
                    if perform_intersection and intersects_logical:
                        geometry_fill[idx] = current_geometry.intersection(subset_geom)

            if perform_intersection:
                if geometry_fill is None:
                    geometry_variable = GeometryVariable(name=geom_name)
                else:
                    geometry_variable = GeometryVariable(name=geom_name, value=geometry_fill, mask=fill_mask,
                                                         dimensions=ret.element_dim)
                ret.parent.add_variable(geometry_variable, force=True)

            the_slice = np.invert(fill_mask)
            if apply_slice:
                sliced_obj = ret.get_distributed_slice(the_slice)
            else:
                sliced_obj = ret
            sliced_mask_value = sliced_obj.get_mask()

            # Only modify the outgoing mask if any values are masked.
            if sliced_mask_value is not None and sliced_mask_value.any():
                sliced_obj.set_mask(sliced_mask_value, cascade=cascade)

        # The element dimension needs to be updated to account for fancy slicing which may leave some ranks empty.
        new_element_dimension_name = self.element_dim.name
        if sliced_obj.is_empty:
            new_element_dimension_size = 0
            new_element_dimension_src_idx = None
        else:
            element_dim = sliced_obj.element_dim
            new_element_dimension_size = element_dim.size
            new_element_dimension_src_idx = element_dim._src_idx

        new_element_dimension = create_distributed_dimension(new_element_dimension_size,
                                                             name=new_element_dimension_name,
                                                             src_idx=new_element_dimension_src_idx)
        sliced_obj.parent.dimensions[new_element_dimension.name] = new_element_dimension

        if perform_intersection:
            obj_to_ret = sliced_obj.parent[geometry_variable.name]
        else:
            obj_to_ret = sliced_obj

        if return_slice:
            ret = (obj_to_ret, the_slice)
        else:
            ret = obj_to_ret

        return ret

    def iter_geometries(self, **kwargs):
        for yld in self.get_geometry_iterable(**kwargs):
            yield yld

    def iter_records(self, use_mask=True):
        raise NotImplementedError

    def reduce_global(self):
        """
        De-duplicate and reindex (reset start index) an element node connectivity variable. Operation is collective
        across the current VM. The new node dimension is distributed. Return a shallow copy of `self` for convenience.

        :rtype: :class:`~ocgis.spatial.geomc.AbstractGeometryCoordinates`
        """
        ocgis_lh(msg='entering reduce_global', logger='geomc', level=logging.DEBUG)
        raise_if_empty(self)

        if self.cindex is None:
            raise ValueError('A coordinate index is required to reduce coordinates.')

        ocgis_lh(msg='starting reduce_reindex_coordinate_index', logger='geomc', level=logging.DEBUG)
        new_cindex, uidx = reduce_reindex_coordinate_index(self.cindex, start_index=self.start_index)
        ocgis_lh(msg='finished reduce_reindex_coordinate_index', logger='geomc', level=logging.DEBUG)

        new_cindex = Variable(name=self.cindex.name, value=new_cindex, dimensions=self.cindex.dimensions)

        ret = self.copy()
        if self.start_index == 1:
            uidx -= 1
        new_parent = self.x[uidx].parent

        cdim = new_parent[self.x.name].dimensions[0]
        new_node_dimension = create_distributed_dimension(cdim.size, name=cdim.name, src_idx=cdim._src_idx)
        new_parent.dimensions[cdim.name] = new_node_dimension

        new_parent[self.cindex.name].extract(clean_break=True)
        ret.parent = new_parent
        ret.cindex = new_cindex
        ocgis_lh(msg='exiting reduce_global', logger='geomc', level=logging.DEBUG)
        return ret

    def _get_dimensions_(self):
        return tuple([self.element_dim])

    def _get_extent_(self):
        # Get the x and y coordinate values.
        x, y = self.x.v(), self.y.v()
        return x.min(), y.min(), x.max(), y.max()

    def _get_is_empty_(self):
        return self.parent.is_empty

    @staticmethod
    def _gc_create_global_indices_(*args, **kwargs):
        return None

    def _gc_initialize_(self, regridding_role):
        if regridding_role == RegriddingRole.SOURCE:
            name = GridChunkerConstants.IndexFile.NAME_SRCIDX_GUID
        elif regridding_role == RegriddingRole.DESTINATION:
            name = GridChunkerConstants.IndexFile.NAME_DSTIDX_GUID
        else:
            raise NotImplementedError(regridding_role)
        element_dim = self.element_dim
        src_index = arange_from_dimension(element_dim, start=1, dtype=env.NP_INT)
        src_index_var = Variable(name=name, value=src_index, dimensions=element_dim)
        self.parent.add_variable(src_index_var)

    def _gc_nchunks_dst_(self, grid_chunker):
        try:
            ret = super(AbstractGeometryCoordinates, self)._gc_nchunks_dst_(grid_chunker)
        except NotImplementedError:
            ret = (100,)
        return ret

    def _initialize_parent_(self, *args, **kwargs):
        return self._get_parent_class_()(*args, **kwargs)


class PointGC(AbstractGeometryCoordinates):
    abstraction = GridAbstraction.POINT
    __shapely_geometry_class__ = Point
    __use_bounds_intersects_optimizations__ = True

    @staticmethod
    def get_element_node_connectivity_by_index(element_connectivity, idx):
        idx = element_connectivity[idx, ...].flatten()[0]
        return idx


class LineGC(AbstractGeometryCoordinates):
    abstraction = GridAbstraction.LINE
    __shapely_geometry_class__ = LineString
    __use_bounds_intersects_optimizations__ = False

    def __init__(self, *args, **kwargs):
        raise RequestableFeature

    @staticmethod
    def get_element_node_connectivity_by_index(element_connectivity, idx):
        raise NotImplementedError


class PolygonGC(AbstractGeometryCoordinates):
    abstraction = GridAbstraction.POLYGON
    __shapely_geometry_class__ = Polygon
    __shapely_multipart_class__ = MultiPolygon
    __use_bounds_intersects_optimizations__ = False

    def get_element_node_connectivity_by_index(self, element_connectivity, idx):
        # TODO: OPTIMIZE: Driver-specific method to load polygon coordinate indices.
        # ESMF unstructured uses counts...
        if self.dimension_map.get_driver() == DriverKey.NETCDF_ESMF_UNSTRUCT:
            num_element_conn_value = self.parent['numElementConn'].get_value()
            if idx == 0:
                start = 0
            else:
                start = num_element_conn_value[0:idx].sum()
            stop = num_element_conn_value[0:idx + 1].sum()
            ret = element_connectivity[start:stop]
        else:
            ret = element_connectivity[idx, ...].flatten()
            if element_connectivity.dtype == object:
                ret = ret[0].flatten()
        # TODO: /OPTIMIZE
        return ret

    def get_shapely_geometry(self, *args, **kwargs):
        if len(args) == 3:
            x, y, z = args
        else:
            x, y = args
            z = None

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        if z is None:
            stup = (x, y)
        else:
            z = z.reshape(-1, 1)
            stup = (x, y, z)
        c = np.hstack(stup)
        return self.__shapely_geometry_class__(c, **kwargs)


def create_buffer_array(value=MPI_EMPTY_VALUE, dtype='i'):
    return np.array([value], dtype=dtype)


def create_Irecv(source, tag, dtype='i', mtype=None):
    if mtype is None:
        mtype = vm.get_mpi_type(np.int32)

    buf = create_buffer_array(dtype=dtype)
    req = vm.comm.Irecv([buf, mtype], source=source, tag=tag)
    return buf, req


def create_Irecv_dct(ranks, tag):
    ret = {}
    for rank in ranks:
        data, req = create_Irecv(rank, tag)
        ret[rank] = {'data': data, 'req': req}
    return ret


def create_Isend(dest, tag, value=MPI_EMPTY_VALUE, dtype='i', mtype=None):
    from mpi4py import MPI
    if mtype is None:
        mtype = MPI.INT

    buf = create_buffer_array(value=value, dtype=dtype)
    req = vm.comm.Isend([buf, mtype], dest=dest, tag=tag)
    return buf, req


def get_default_geometry_variable_name(gc):
    possible = {GridAbstraction.POINT: VariableName.GEOMETRY_POINT,
                GridAbstraction.LINE: VariableName.GEOMETRY_LINE,
                GridAbstraction.POLYGON: VariableName.GEOMETRY_POLYGON}
    return possible[gc.abstraction]


def get_xyz_select(x, y, bounds, z=None, z_bounds=None, invert=False, keep_touches=True):
    from ocgis.spatial.grid import arr_intersects_bounds

    minx, miny, maxx, maxy = bounds
    select_x = arr_intersects_bounds(x, minx, maxx, keep_touches=keep_touches)
    select_y = arr_intersects_bounds(y, miny, maxy, keep_touches=keep_touches)
    select = np.logical_and(select_x, select_y)
    if z_bounds is not None:
        minz, maxz = z_bounds
        select_z = arr_intersects_bounds(z, minz, maxz, keep_touches=keep_touches)
        select = np.logical_and(select, select_z)
    if invert:
        select = np.invert(select)
    return select


def iter_multipart_coordinates(arr, mbv):
    """
    Split an array by a multi-part break value generating the sections that when combined create the original array.

    :param arr: Array containing multi-part break values (it doesn't have to contain them).
    :type arr: :class:`numpy.ndarray`
    :param int mbv: The multi-part break value. Typically this is a negative integer.
    :return: :class:`numpy.ndarray`
    """
    w = np.where(arr == mbv)[0]
    l = len(arr)
    w = np.hstack((w, [l]))
    start = 0
    for idx, iw in enumerate(w):
        slc = slice(start, iw)
        start = iw + 1
        yld = arr.__getitem__(slc)
        yield yld


def reduce_reindex_coordinate_index(cindex, start_index=0):
    """
    Reindex a subset of global coordinate indices contained in the ``cindex`` variable.

    The starting index value (``0`` or ``1``) is set by ``start_index`` for the re-indexing procedure.

    Function will not respect masks.

    The function returns a two-element tuple:

     * First element --> A :class:`numpy.ndarray` with the same dimension as ``cindex`` containing the new indexing.
     * Second element --> A :class:`numpy.ndarray` containing the unique indices that may be used to reduce an external
       coordinate storage variable or array.

    :param cindex: A variable containing coordinate index integer values. This variable may be distributed. This may
     also be a NumPy array.
    :type cindex: :class:`~ocgis.Variable` | :class:`~numpy.ndarray`
    :param int start_index: The first index to use for the re-indexing of ``cindex``. This may be ``0`` or ``1``.
    :rtype: tuple
    """
    ocgis_lh(msg='entering reduce_reindex_coordinate_index', logger='geomc', level=logging.DEBUG)

    # Get the coordinate index values as a NumPy array.
    try:

        ocgis_lh(msg='calling cindex.get_value()', logger='geomc', level=logging.DEBUG)
        ocgis_lh(msg='cindex.has_allocated_value={}'.format(cindex.has_allocated_value), logger='geomc',
                 level=logging.DEBUG)
        ocgis_lh(msg='cindex.dimensions[0]={}'.format(cindex.dimensions[0]), logger='geomc', level=logging.DEBUG)
        cindex = cindex.get_value()
        ocgis_lh(msg='finished cindex.get_value()', logger='geomc', level=logging.DEBUG)
    except AttributeError:
        # Assume this is already a NumPy array.
        pass

    # Only work with 1D arrays.
    cindex = np.atleast_1d(cindex)
    # Used to return the coordinate index to the original shape of the incoming coordinate index.
    original_shape = cindex.shape
    cindex = cindex.flatten()

    # Create the unique coordinate index array.
    ocgis_lh(msg='calling create_unique_global_array', logger='geomc', level=logging.DEBUG)
    if vm.size > 1:
        u = np.array(create_unique_global_array(cindex))
    else:
        u = np.unique(cindex)
    ocgis_lh(msg='finished create_unique_global_array', logger='geomc', level=logging.DEBUG)

    # Synchronize the data type for the new coordinate index.
    lrank = vm.rank
    if lrank == 0:
        dtype = u.dtype
    else:
        dtype = None
    dtype = vm.bcast(dtype)

    # Flag to indicate if the current rank has any unique values.
    has_u = len(u) > 0

    # Create the new coordinate index.
    new_u_dimension = create_distributed_dimension(len(u), name='__new_u_dimension__')
    new_u = arange_from_dimension(new_u_dimension, start=start_index, dtype=dtype)

    # Create a hash for the new index. This is used to remap the old coordinate index.
    if has_u:
        uidx = {ii: jj for ii, jj in zip(u, new_u)}
    else:
        uidx = None

    vm.barrier()

    # Construct local bounds for the rank's unique value. This is used as a cheap index when ranks are looking for
    # index overlaps.
    if has_u:
        local_bounds = min(u), max(u)
    else:
        local_bounds = None
    # Put a copy for the bounds indexing on each rank.
    lb_global = vm.gather(local_bounds)
    lb_global = vm.bcast(lb_global)

    # Find the vm ranks the local rank cares about. It cares if unique values have overlapping unique bounds.
    overlaps = []
    for rank, lb in enumerate(lb_global):
        if rank == lrank:
            continue
        if lb is not None:
            contains = lb[0] <= cindex
            contains = np.logical_and(lb[1] >= cindex, contains)
            if np.any(contains):
                overlaps.append(rank)

    # Ranks must be able to identify which ranks will be asking them for data.
    global_overlaps = vm.gather(overlaps)
    global_overlaps = vm.bcast(global_overlaps)
    destinations = [ii for ii, jj in enumerate(global_overlaps) if vm.rank in jj]

    # MPI communication tags used in the algorithm.
    tag_search = MPITag.REDUCE_REINDEX_SEARCH
    tag_success = MPITag.REDUCE_REINDEX_SUCCESS
    tag_child_finished = MPITag.REDUCE_REINDEX_CHILD_FINISHED
    tag_found = MPITag.REDUCE_REINDEX_FOUND

    # Fill array for the new coordinate index.
    new_cindex = np.empty_like(cindex)

    # vm.barrier_print('starting run_rr')
    # Fill the new coordinate indexing.
    if lrank == 0:
        run_rr_root(new_cindex, cindex, uidx, destinations, tag_child_finished, tag_found, tag_search, tag_success)
    else:
        run_rr_nonroot(new_cindex, cindex, uidx, destinations, has_u, overlaps, tag_child_finished, tag_found,
                       tag_search,
                       tag_success)
    # vm.barrier_print('finished run_rr')

    # Return array to its original shape.
    new_cindex = new_cindex.reshape(*original_shape)

    vm.barrier()

    return new_cindex, u


def run_rr_nonroot(new_cindex, cindex, uidx, destinations, has_u, overlaps, tag_child_finished, tag_found, tag_search,
                   tag_success):
    # Fill each value in the coordinate index.
    for idx, ii in enumerate(cindex.flat):
        # if idx % 100 == 0:
        #     vm.rank_print('{} of {}'.format(idx+1, cindex.shape[0]))
        try:
            # If this rank has unique indices, try to retrieve the new indexing from its unique indexing hash.
            if has_u:
                new_cindex_value = uidx[ii]
            # Jump into the search loop if there are no unique indices on the local rank.
            else:
                raise KeyError
        except KeyError:
            new_cindex_value = None
            # Keep searching and waiting for the response from the overlap ranks.
            # while new_cindex_value is None:

            # Send the value to find to each of the destination ranks.
            assert ii != MPI_EMPTY_VALUE
            assert ii >= 0
            # vm.rank_print('idx', idx, 'sending to:', overlaps, 'receiving from:', destinations)
            search_reqs = [create_Isend(orank, tag_search, value=ii) for orank in overlaps]
            # for s in search_reqs:
            #     s.Test()

            # vm.rank_print('receiving from:', destinations)
            sent = search_for_destinations(destinations, uidx, tag_found, tag_search)

            for s in search_reqs:
                # vm.rank_print('waiting for search request', 'idx', idx, s[0])
                s[1].wait()

            # vm.rank_print('overlaps', overlaps, 'destinations', destinations)
            # time.sleep(100)

            for overlap_rank in overlaps:
                data, req = create_Irecv(overlap_rank, tag_found)
                req.wait()
                # if req.Test():
                if data[0] != MPI_EMPTY_VALUE:
                    new_cindex_value = data[0]

            for s in sent:
                s[1].wait()

                # for s in search_reqs:
                #     s.wait()

                # Free the search requests to avoid any race conditions in data buffers.

                # cancel_free_requests(search_reqs)

        # Fill the new coordinate index array with the found value.
        assert new_cindex_value is not None
        new_cindex[idx] = new_cindex_value

    # Continue searching for destination ranks until the success signal is received from the root rank.
    _, req_child_finished = create_Isend(0, tag_child_finished)
    _, req_success = create_Irecv(0, tag_success)

    while not req_success.Test():
        sent = search_for_destinations(destinations, uidx, tag_found, tag_search)
        for s in sent:
            s[1].wait()
    # Wait until the child finished tag is received by the parent.

    req_child_finished.wait()


def run_rr_root(new_cindex, cindex, uidx, destinations, tag_child_finished, tag_found, tag_search, tag_success):
    # Tracks when ranks are finished.
    children_finished = [False] * vm.size
    children_finished[0] = True

    # Fill the new coordinate index. The root rank will always fully map itself.
    for idx, ii in enumerate(cindex.flat):
        new_cindex[idx] = uidx[ii]
    success = False

    # Open channels for child finished signals.
    children_finished_reqs = create_Irecv_dct(vm.ranks, tag_child_finished)

    while not success:
        # Check for any requests from destination ranks. These ranks need a value that may be on this rank.
        sent = search_for_destinations(destinations, uidx, tag_found, tag_search)
        for s in sent:
            s[1].wait()

        # Check if children have finished. If they have finished. Send the success signal to other participating ranks.
        for idx, rank in enumerate(vm.ranks):
            if not children_finished[idx]:
                req_child_finished = children_finished_reqs[rank]['req']
                if req_child_finished.Test():
                    children_finished[rank] = True
        if all(children_finished):
            success = True
            for rank in vm.ranks:
                if rank != 0:
                    _, req = create_Isend(rank, tag_success)
                    req.wait()


def search_for_destinations(dest_ranks, uidx, tag_found, tag_search):
    assert isinstance(dest_ranks, list)
    sent = []
    for dest_rank in dest_ranks:
        data, req = create_Irecv(dest_rank, tag_search)
        # data = np.array([MPI_EMPTY_VALUE], dtype='i')
        # buf = [data, MPI.INT]
        # req = vm.comm.Irecv(buf, source=dest_rank, tag=tag_search)
        # req.wait()
        if req.Test():
            search_value = data[0]
            # if data == MPI_EMPTY_VALUE:
            #     print('rank=', vm.rank, 'bad data from rank:', dest_rank)
            assert search_value != MPI_EMPTY_VALUE
            assert search_value is not None
            assert search_value >= 0
            # vm.rank_print('search_value', search_value)
            local_uidx = uidx.get(search_value, MPI_EMPTY_VALUE)
            send_res = create_Isend(dest_rank, tag_found, value=local_uidx)
            sent.append(send_res)
        else:
            cancel_free_requests([req])
    return sent
