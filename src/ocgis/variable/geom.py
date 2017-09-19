from collections import deque
from copy import deepcopy
from itertools import product

import numpy as np
from numpy.core.multiarray import ndarray
from shapely import wkb
from shapely.geometry import Point, Polygon, MultiPolygon, mapping, MultiPoint, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import cascaded_union
from shapely.prepared import prep

from ocgis import Variable, vm
from ocgis import constants
from ocgis import env
from ocgis.base import AbstractOcgisObject
from ocgis.base import get_dimension_names, get_variable_names, raise_if_empty
from ocgis.constants import KeywordArgument, HeaderName, VariableName, DimensionName, ConversionTarget, DriverKey
from ocgis.environment import ogr
from ocgis.exc import EmptySubsetError, RequestableFeature
from ocgis.spatial.base import AbstractSpatialVariable
from ocgis.util.helpers import iter_array, get_trimmed_array_by_mask, get_swap_chain, find_index
from ocgis.variable.base import get_dimension_lengths
from ocgis.variable.crs import Cartesian
from ocgis.variable.dimension import create_distributed_dimension, Dimension
from ocgis.variable.iterator import Iterator

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

GEOM_TYPE_MAPPING = {'Polygon': Polygon, 'Point': Point, 'MultiPoint': MultiPoint, 'MultiPolygon': MultiPolygon}


class GeometryProcessor(AbstractOcgisObject):
    """
    :param geometry_iterable: Yields a Shapely geometry object or ``None``. This may yield ``None`` to assist in index
     tracking. For example, in a two-dimensional array of geometry objects a significant portion of these may be masked
     before a more complex subset operation.
    :param subset_geometry: The geometry used to subset ``geometry_iterable``.
    :param keep_touches: If ``True``, keep geometries that only touch the subset geometry.
    """

    def __init__(self, geometry_iterable, subset_geometry, keep_touches=False):
        self.geometry_iterable = geometry_iterable
        self.subset_geometry = subset_geometry
        self.keep_touches = keep_touches

        self._is_used = False

    def iter_intersection(self):
        """
        Yields a tuple similar to :meth:`ocgis.new_interface.geom.GeometryProcessor.iter_intersects`. However, if the
        current geometry does not intersect ``subset_geometry``, the geometry is ``None`` as opposed to a value.

        :return: tuple
        :raises: ValueError
        """
        for idx, intersects_logical, geometry in self.iter_intersects():
            if intersects_logical:
                geometry = geometry.intersection(self.subset_geometry)
            else:
                geometry = None
            yield idx, intersects_logical, geometry

    def iter_intersects(self):
        """
        Yields the enumerated index and ``True`` if the current geometry intersects ``subset_geometry``. ``False``
        otherwise. If the current geometry is ``None``, then ``False`` is always returned. The current geometry is also
        yielded.

        An example yielded tuple: ``(0, False, <Point>)``

        :return: tuple
        :raises: ValueError
        """

        if self._is_used:
            raise ValueError('Iterator already used. Please re-initialize.')
        else:
            self._is_used = True

        subset_geometry = self.subset_geometry
        keep_touches = self.keep_touches
        prepared = prep(subset_geometry)
        prepared_intersects = prepared.intersects
        subset_geometry_touches = subset_geometry.touches

        for idx, geometry in self.geometry_iterable:
            yld = False
            # If the yielded geometry is None, then it should not be considered within the subset geometry.
            if geometry is not None:
                if prepared_intersects(geometry):
                    yld = True
                    if not keep_touches and subset_geometry_touches(geometry):
                        yld = False
            yield idx, yld, geometry


class GeometryVariable(AbstractSpatialVariable):
    """
    A variable containing Shapely geometry object arrays.
    
    .. note:: Accepts all parameters to :class:`~ocgis.Variable`.

    Additional keyword arguments are:

    :param crs: (``=None``) The coordinate reference system for the geometries.
    :type crs: :class:`~ocgis.variable.crs.AbstractCRS`
    :param str geom_type: (``='auto'``) See http://toblerity.org/shapely/manual.html#object.geom_type. If ``'auto'``,
     the geometry type will be automatically determined from the object array. Providing a default prevents iterating
     over the object array to identify the geometry type.
    :param ugid: (``=None``) An integer array with same shape as the geometry variable. This array will be converted to
     a :class:`~ocgis.Variable`.
    :type ugid: :class:`numpy.ndarray`
    :param bool is_bbox: If ``True``, treat the polygon geometry as a bounding box geometry.
    """

    def __init__(self, **kwargs):
        kwargs = kwargs.copy()
        self._geom_type = kwargs.pop(KeywordArgument.GEOM_TYPE, 'auto')

        if kwargs.get(KeywordArgument.NAME) is None:
            kwargs[KeywordArgument.NAME] = VariableName.GEOMETRY_VARIABLE

        ugid = kwargs.pop(KeywordArgument.UGID, None)
        self.is_bbox = kwargs.pop('is_bbox', False)
        self.is_prepared = kwargs.pop('is_prepared', False)

        super(GeometryVariable, self).__init__(**kwargs)

        if ugid is not None:
            ugid_var = Variable(name=HeaderName.ID_SELECTION_GEOMETRY, value=[ugid], dimensions=self.dimensions)
            self.set_ugid(ugid_var)

    @property
    def area(self):
        """
        :return: geometry areas as a float masked array
        :rtype: :class:`numpy.ma.MaskedArray`
        """
        if self.is_empty:
            fill = None
        else:
            r_value = self.get_masked_value()
            fill = np.ones(r_value.shape, dtype=env.NP_FLOAT)

            mask = self.get_mask()
            if mask is not None:
                mask = mask.copy()

            fill = np.ma.array(fill, mask=mask)
            for slc, geom in iter_array(r_value, return_value=True):
                fill.data[slc] = geom.area
        return fill

    @property
    def dtype(self):
        """
        :return: geometry variables are always of object type
        :rtype: type
        """
        # Geometry arrays are always object arrays.
        return object

    @dtype.setter
    def dtype(self, value):
        # Geometry data types are always objects. Ignore any passed value.
        pass

    @property
    def geom_type(self):
        """
        :return: geometry type for the variable
        :rtype: str
        """
        # Geometry objects may change part counts during operations. It is better to scan and update the geometry types
        # to account for these operations.
        if self._geom_type == 'auto':
            self._geom_type = get_geom_type(self.get_value())
        return self._geom_type

    @property
    def geom_type_global(self):
        """
        :returns: global geometry type collective across the current :class:`~ocgis.OcgVM`
        :rtype: str
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """
        raise_if_empty(self)
        geom_types = vm.gather(self.geom_type)
        if vm.rank == 0:
            for g in geom_types:
                if g.startswith('Multi'):
                    break
        else:
            g = None
        return vm.bcast(g)

    @property
    def has_z(self):
        """
        Return ``True`` if the variable has a z-coordinate.

        :rtype: bool
        """

        if self.is_empty:
            ret = False
        else:
            ret = self.get_masked_value().compressed()[0].has_z
        return ret

    @property
    def weights(self):
        """
        Weights are defined as:

        >>> self.area / self.area.max()

        Any geometries with zero area (points) are given an area of ``1.0`` for the purposes of weight calculation.

        :return:  weights as a float masked array
        :rtype: :class:`numpy.ma.MaskedArray`
        """

        area = self.area
        area.data[area.data == 0] = 1.0
        return area / area.max()

    def as_shapely(self):
        """
        Convert to a Shapely geometry provided this is a singleton geometry variable.

        :rtype: :class:`shapely.geometry.base.BaseGeometry`
        """
        f = self.get_value().flatten()
        assert f.size == 1
        return f[0]

    def convert_to(self, target=ConversionTarget.GEOMETRY_COORDS, **kwargs):
        """
        Convert to a target type. Some common manipulations are shared between conversion targets:

        * Always orients polygons CCW.

        :param target: The target type.
        :type target: :attr:`ocgis.constants.ConversionTarget`
        :keyword dtype: ``(=None)`` Array data type for the coordinate variables.
        :keyword xname: ``(=constants.DEFAULT_NAME_COL_COORDINATES)`` Name of the x-coordinate variable.
        :keyword yname: ``(=constants.DEFAULT_NAME_ROW_COORDINATES)`` Name of the y-coordinate variable.
        :keyword zname: ``(=constants.DEFAULT_NAME_LVL_COORDINATES)`` Name of the z-coordinate variable.
        :keyword node_dim_name: ``(='n_node')`` Name of the node count dimension.
        :keyword element_index_name: ``(='element_index')`` Name of the element node connectivity variable.
        :keyword pack: ``(=True)`` If ``True``, de-duplicate coordinate vectors.
        :keyword repeat_last_node: ``(=False)`` If ``False``, do not repeat the last node coordinate for polygon
         geometries.
        :keyword max_element_coords: ``(=None)`` If provided, the maximum number of coordinates across all polygon
         geometries to convert. This fixes the column count for element node connectivity arrays. Otherwise, ragged
         arrays are used.
        :type max_element_coords: int
        """

        # TODO: This needs to handle multi-geometries.
        # TODO: This needs to handle geometry holes/interiors.
        # TODO: Implement line conversion.

        from ocgis.spatial.geomc import PointGC, PolygonGC

        assert self.ndim == 1

        kwargs = kwargs.copy()
        dtype = kwargs.pop(KeywordArgument.DTYPE, None)
        xname = kwargs.pop('xname', constants.DEFAULT_NAME_COL_COORDINATES)
        yname = kwargs.pop('yname', constants.DEFAULT_NAME_ROW_COORDINATES)
        zname = kwargs.pop('zname', constants.DEFAULT_NAME_LVL_COORDINATES)
        node_dim_name = kwargs.pop('node_dim_name', 'n_node')
        element_index_name = kwargs.pop('element_index_name', 'element_index')
        pack = kwargs.pop('pack', True)
        repeat_last_node = kwargs.pop('repeat_last_node', False)
        max_element_coords = kwargs.pop('max_element_coords', None)
        assert len(kwargs) == 0

        geom_type = self.geom_type
        if target == ConversionTarget.GEOMETRY_COORDS:
            has_z = self.has_z
            geom_itr = self.get_value().flat
            if geom_type == 'Point':
                if has_z:
                    ndim = 3
                else:
                    ndim = 2
                fill = np.zeros([self.size, ndim], dtype=dtype)
                for idx, geom in enumerate(geom_itr):
                    fill[idx] = np.array(geom)
                xv = fill[:, 0]
                yv = fill[:, 1]
                if has_z:
                    zv = fill[:, 2]
                else:
                    zv = None
            elif geom_type == 'Polygon':
                if max_element_coords is not None:
                    element_index = np.zeros((self.size, max_element_coords), dtype=env.NP_INT)
                else:
                    element_index = np.zeros(self.size, dtype=object)
                cidx = 0
                xv = deque()
                yv = deque()
                zv = deque()
                seqs = [xv, yv]
                if has_z:
                    seqs.append(zv)
                for idx, geom in enumerate(geom_itr):
                    geom = get_ccw_oriented_and_valid_shapely_polygon(geom)
                    coords = np.array(geom.exterior.coords)
                    if repeat_last_node:
                        coords_shape = coords.shape[0]
                    else:
                        coords_shape = coords.shape[0] - 1
                    if pack:
                        curr_element_index = np.zeros(coords_shape, dtype=env.NP_INT)
                        for coords_row_idx in range(coords_shape):
                            coords_row = coords[coords_row_idx, :].flatten()
                            found_index = find_index(seqs, coords_row)
                            if found_index is None:
                                xv.append(coords_row[0])
                                yv.append(coords_row[1])
                                if has_z:
                                    zv.append(coords_row[2])
                                found_index = cidx
                                cidx += 1
                            curr_element_index[coords_row_idx] = found_index
                        element_index[idx] = curr_element_index
                    else:
                        xv.extend(coords[0:coords_shape, 0].flatten().tolist())
                        yv.extend(coords[0:coords_shape, 1].flatten().tolist())
                        if has_z:
                            zv.extend(coords[0:coords_shape, 2].flatten().tolist())
                        fill = np.arange(cidx, cidx + coords_shape, dtype=env.NP_INT)
                        if max_element_coords is None:
                            element_index[idx] = fill
                        else:
                            element_index[idx, :] = fill
                        cidx += coords_shape
                if max_element_coords is None:
                    element_index_dims = self.dimensions[0]
                else:
                    element_index_dims = [self.dimensions[0],
                                          Dimension(name=DimensionName.UGRID_MAX_ELEMENT_COORDS,
                                                    size=max_element_coords)]
                element_index = Variable(name=element_index_name, value=element_index, dimensions=element_index_dims)
            else:
                msg = "Conversion for this geometry type is not implemented: '{}'".format(geom_type)
                raise RequestableFeature(message=msg)

            names = [xname, yname, zname]
            values = [xv, yv, zv]
            variables = [None] * 3

            if geom_type == 'Point':
                dim = self.dimensions[0]
            elif geom_type == 'Polygon':
                dim = create_distributed_dimension(len(xv), name=node_dim_name)
            else:
                raise NotImplementedError(geom_type)

            for idx in range(len(names)):
                if not has_z and idx == 2:
                    break
                variables[idx] = Variable(name=names[idx], value=values[idx], dimensions=dim)

            if geom_type == 'Point':
                klass = PointGC
            elif geom_type == 'Polygon':
                klass = PolygonGC
            else:
                raise NotImplementedError(geom_type)

            kwds = dict(z=variables[2], crs=self.crs)
            if geom_type == 'Polygon':
                kwds['cindex'] = element_index
                kwds['packed'] = pack
            if self.has_mask:
                kwds[KeywordArgument.MASK] = self.get_mask()
            ret = klass(variables[0], variables[1], **kwds)

        else:
            raise NotImplementedError(target)
        return ret

    @classmethod
    def from_shapely(cls, geom, **kwargs):
        kwargs = kwargs.copy()
        kwargs[KeywordArgument.VALUE] = [geom]
        if KeywordArgument.DIMENSIONS not in kwargs:
            kwargs[KeywordArgument.DIMENSIONS] = DimensionName.GEOMETRY_DIMENSION
        return cls(**kwargs)

    def get_buffer(self, *args, **kwargs):
        """
        Return a shallow copy of the geometry variable with geometries buffered.

        .. note:: Accepts all parameters to :meth:`shapely.geometry.base.BaseGeometry.buffer`.

        An additional keyword argument is:

        :keyword str geom_type: The geometry type for the new buffered geometry if known in advance.
        :rtype: :class:`~ocgis.GeometryVariable`
        :raises: :class:`~ocgis.exc.EmptyObjectError`
        """

        raise_if_empty(self)

        # New geometry type for the buffered object.
        geom_type = kwargs.pop('geom_type', 'auto')

        ret = self.copy()
        new_value = np.empty_like(ret.get_value(), dtype=object)
        to_buffer = self.get_value()
        mask = self.get_mask()
        for idx, mask_value in iter_array(mask, return_value=True):
            if not mask_value:
                new_value[idx] = to_buffer[idx].buffer(*args, **kwargs)
            else:
                new_value[idx] = None

        ret.set_value(new_value)
        ret._geom_type = geom_type

        return ret

    def get_intersects(self, *args, **kwargs):
        """
        Perform an intersects spatial operations on the geometry variable.

        :keyword bool return_slice: (``=False``) If ``True``, return the _global_ slice that will guarantee no masked
         elements outside the subset geometry as the second element in the return value.
        :keyword bool cascade: (``=True``) If ``True`` (the default), set the mask following the spatial operation on
         all variables in the parent collection.
        :returns: shallow copy of the geometry variable
        :rtype: :class:`~ocgis.GeometryVariable` | ``(<geometry variable>, <slice>)``
        :raises: :class:`~ocgis.exc.EmptySubsetError`
        """

        raise_if_empty(self)

        return_slice = kwargs.pop(KeywordArgument.RETURN_SLICE, False)
        cascade = kwargs.pop(KeywordArgument.CASCADE, True)

        ret = self.copy()
        intersects_mask_value = ret.get_mask_from_intersects(*args, **kwargs)
        ret, ret_mask, ret_slice = get_masking_slice(intersects_mask_value, ret)

        if not ret.is_empty:
            ret.set_mask(ret_mask.get_value(), cascade=cascade, update=True)
        else:
            for var in list(ret.parent.values()):
                assert var.is_empty

        # TODO: need to implement fancy index-based slicing for the one-dimensional unstructured case. Difficult in parallel.
        # if self.ndim == 1:
        #     # For one-dimensional data, assume it is unstructured and compress the returned data.
        #     adjust = np.where(np.invert(ret.get_mask()))
        #     ret_slc = adjust

        if return_slice:
            ret = (ret, ret_slice)

        return ret

    def get_intersection(self, *args, **kwargs):
        """
        .. note:: Accepts all parameters to :meth:`~ocgis.new_interface.geom.GeometryVariable.get_intersects`. Same
         return types.

        Additional arguments and/or keyword arguments are:

        :keyword bool inplace: (``=False``) If ``False`` (the default), deep copy the geometry array on the output
         before executing an intersection. If ``True``, modify the geometries in-place.
        :keyword bool intersects_check: (``=True``) If ``True`` (the default), first perform an intersects operation to
         limit the geometries tests for intersection. If ``False``, perform the intersection as is.
        """

        inplace = kwargs.pop(KeywordArgument.INPLACE, False)
        intersects_check = kwargs.pop(KeywordArgument.INTERSECTS_CHECK, True)
        return_slice = kwargs.get(KeywordArgument.RETURN_SLICE, False)
        subset_geometry = args[0]

        if intersects_check:
            ret = self.get_intersects(*args, **kwargs)
        else:
            if inplace:
                ret = self
            else:
                ret = self.copy()

        if intersects_check:
            # If indices are being returned, this will be a tuple.
            if return_slice:
                obj = ret[0]
            else:
                obj = ret
        else:
            if return_slice:
                global_slice = [(slice(d.bounds_global[0], d.bounds_global[1]) for d in self.dimensions)]
                ret = (ret, global_slice)
                obj = ret
            else:
                obj = ret

        if not obj.is_empty:
            if not inplace:
                obj.set_value(deepcopy(obj.get_value()))
            obj_value = obj.get_masked_value()
            for idx, geom in iter_array(obj_value, return_value=True):
                obj_value.data[idx] = geom.intersection(subset_geometry)
        return ret

    def get_iter(self, *args, **kwargs):
        """
        :rtype: :class:`~ocgis.variable.iterator.Iterator`
        """
        should_add = kwargs.pop(KeywordArgument.ADD_GEOM_UID, False)

        if should_add and self.ugid is not None:
            followers = [self.ugid]
        else:
            followers = []

        return Iterator(self, followers=followers, **kwargs)

    def get_mask_from_intersects(self, geometry_or_bounds, use_spatial_index=env.USE_SPATIAL_INDEX, keep_touches=False,
                                 original_mask=None):
        """
        :param geometry_or_bounds: A Shapely geometry or bounds tuple used for the masking.
        :type geometry_or_bounds: :class:`shapely.geometry.base.BaseGeometry` | :class:`tuple`
        :param bool use_spatial_index: If ``True``, use a spatial index for the operation.
        :param bool keep_touches: If ``True``, keep geometries that only touch.
        :param original_mask: A hint mask for the spatial operation. ``True`` values will be skipped.
        :type original_mask: :class:`numpy.ndarray`
        :returns: boolean array with non-intersecting values set to ``True``
        :rtype: :class:`numpy.ndarray`
        """
        raise_if_empty(self)

        # Transform bounds sequence to a geometry.
        if not isinstance(geometry_or_bounds, BaseGeometry):
            geometry_or_bounds = box(*geometry_or_bounds)

        ret = geometryvariable_get_mask_from_intersects(self, geometry_or_bounds,
                                                        use_spatial_index=use_spatial_index,
                                                        keep_touches=keep_touches,
                                                        original_mask=original_mask)
        return ret

    def get_nearest(self, target, return_indices=False):
        """
        :param target: The Shapely geometry to use for proximity.
        :type target: :class:`shapely.geometry.base.BaseGeometry`
        :param bool return_indices: If ``True``, also return the indices used for slicing the geometry variable.
        :return: shallow copy of the geometry variable and optionally slices
        :rtype: :class:`~ocgis.GeometryVariable` | ``(<geometry variable>, <slice>)``
        """
        target = target.centroid
        distances = {}
        for select_nearest_index, geom in iter_array(self.get_value(), return_value=True, mask=self.get_mask()):
            distances[target.distance(geom)] = select_nearest_index
        select_nearest_index = distances[min(distances.keys())]
        ret = self[select_nearest_index]

        if return_indices:
            ret = (ret, select_nearest_index)

        return ret

    def get_report(self):
        if self.crs is None:
            projection = 'NA (no coordinate system)'
            sref = projection
        else:
            projection = self.crs.sr.ExportToProj4()
            sref = self.crs.__class__.__name__

        lines = ['Spatial Reference = {0}'.format(sref),
                 'Proj4 String = {0}'.format(projection),
                 'Geometry Type = {0}'.format(self.geom_type),
                 'Count = {0}'.format(self.size)]

        return lines

    def get_spatial_index(self, target=None):
        """
        :param target: If this is a boolean array, use this as the add target. Otherwise, use the compressed masked
         values.
        :type target: :class:`numpy.ndarray`
        :return: spatial index for the geometry variable
        :rtype: :class:`rtree.index.Index`
        """

        # "rtree" is an optional dependency.
        from ocgis.spatial.index import SpatialIndex
        # Fill the spatial index with unmasked values only.
        si = SpatialIndex()
        # Use compressed masked values if target is not available.
        if target is None:
            target = self.get_masked_value().compressed()
        # Add the geometries to the index.
        r_add = si.add
        for idx, geom in iter_array(target, return_value=True):
            r_add(idx[0], geom)

        return si

    def get_spatial_subset_operation(self, spatial_op, subset_geom, **kwargs):
        if spatial_op == 'intersects':
            ret = self.get_intersects(*[subset_geom], **kwargs)
        elif spatial_op == 'intersection':
            ret = self.get_intersection(*[subset_geom], **kwargs)
        else:
            raise NotImplementedError(spatial_op)
        return ret

    def get_unioned(self, dimensions=None, union_dimension=None, spatial_average=None, root=0):
        """
        Unions _unmasked_ geometry objects. Collective across the current :class:`~ocgis.OcgVM`.
        """
        # TODO: optimize!

        # Get dimension names and lengths for the dimensions to union.
        if dimensions is None:
            dimensions = self.dimensions
        dimension_names = get_dimension_names(dimensions)
        dimension_lengths = [len(self.parent.dimensions[dn]) for dn in dimension_names]

        # Get the variables to spatial average.
        if spatial_average is not None:
            variable_names_to_weight = get_variable_names(spatial_average)
        else:
            variable_names_to_weight = []

        # Get the new dimensions for the geometry variable. The union dimension is always the last dimension.
        if union_dimension is None:
            from ocgis.variable.dimension import Dimension
            union_dimension = Dimension(constants.DimensionName.UNIONED_GEOMETRY, 1)
        new_dimensions = []
        for dim in self.dimensions:
            if dim.name not in dimension_names:
                new_dimensions.append(dim)
        new_dimensions.append(union_dimension)

        # Configure the return variable.
        ret = self.copy()
        if spatial_average is None:
            ret = ret.extract()
        ret.set_mask(None)
        ret.set_value(None)
        ret.set_dimensions(new_dimensions)
        ret.allocate_value()

        # Destination indices in the return variable are filled with non-masked, unioned geometries.
        for dst_indices in product(*[list(range(dl)) for dl in get_dimension_lengths(new_dimensions)]):
            dst_slc = {new_dimensions[ii].name: dst_indices[ii] for ii in range(len(new_dimensions))}

            # Select the geometries to union skipping any masked geometries.
            to_union = deque()
            for indices in product(*[list(range(dl)) for dl in dimension_lengths]):
                dslc = {dimension_names[ii]: indices[ii] for ii in range(len(dimension_names))}
                sub = self[dslc]
                sub_mask = sub.get_mask()
                if sub_mask is None:
                    to_union.append(sub.get_value().flatten()[0])
                else:
                    if not sub_mask.flatten()[0]:
                        to_union.append(sub.get_value().flatten()[0])

            # Execute the union operation.
            processed_to_union = deque()
            for geom in to_union:
                if isinstance(geom, MultiPolygon) or isinstance(geom, MultiPoint):
                    for element in geom:
                        processed_to_union.append(element)
                else:
                    processed_to_union.append(geom)
            unioned = cascaded_union(processed_to_union)

            # Pull unioned geometries and union again for the final unioned geometry.
            if vm.size > 1:
                unioned_gathered = vm.gather(unioned)
                if vm.rank == root:
                    unioned = cascaded_union(unioned_gathered)

            # Fill the return geometry variable value with the unioned geometry.
            to_fill = ret[dst_slc].get_value()
            to_fill[0] = unioned

        # Spatial average shared dimensions.
        if spatial_average is not None:
            # Get source data to weight.
            for var_to_weight in filter(lambda ii: ii.name in variable_names_to_weight, list(self.parent.values())):
                # Holds sizes of dimensions to iterate. These dimension are not squeezed by the weighted averaging.
                range_to_itr = []
                # Holds the names of dimensions to squeeze.
                names_to_itr = []
                # Dimension names that are squeezed. Also the dimensions for the weight matrix.
                names_to_slice_all = []
                for dn in var_to_weight.dimensions:
                    if dn.name in self.dimension_names:
                        names_to_slice_all.append(dn.name)
                    else:
                        range_to_itr.append(len(dn))
                        names_to_itr.append(dn.name)

                # Reference the weights on the source geometry variable.
                weights = self[{nsa: slice(None) for nsa in names_to_slice_all}].weights

                # Path if there are iteration dimensions. Checks for axes ordering in addition.
                if len(range_to_itr) > 0:
                    # New dimensions for the spatially averaged variable. Unioned dimension is always last. Remove the
                    # dimensions aggregated by the weighted average.
                    new_dimensions = [dim for dim in var_to_weight.dimensions if dim.name not in dimension_names]
                    new_dimensions.append(union_dimension)

                    # Prepare the spatially averaged variable.
                    target = ret.parent[var_to_weight.name]
                    target.set_mask(None)
                    target.set_value(None)
                    target.set_dimensions(new_dimensions)
                    target.allocate_value()

                    # Swap weight axes to make sure they align with the target variable.
                    swap_chain = get_swap_chain(dimension_names, names_to_slice_all)
                    if len(swap_chain) > 0:
                        weights = weights.copy()
                    for sc in swap_chain:
                        weights = weights.swapaxes(*sc)

                    # The main weighting loop. Can get quite intensive with many, large iteration dimensions.
                    len_names_to_itr = len(names_to_itr)
                    slice_none = slice(None)
                    squeeze_out = [ii for ii, dim in enumerate(var_to_weight.dimensions) if dim.name in names_to_itr]
                    should_squeeze = True if len(squeeze_out) > 0 else False
                    np_squeeze = np.squeeze
                    np_atleast_1d = np.atleast_1d
                    np_ma_average = np.ma.average
                    for nonweighted_indices in product(*[list(range(ri)) for ri in range_to_itr]):
                        w_slc = {names_to_itr[ii]: nonweighted_indices[ii] for ii in range(len_names_to_itr)}
                        for nsa in names_to_slice_all:
                            w_slc[nsa] = slice_none
                        data_to_weight = var_to_weight[w_slc].get_masked_value()
                        if should_squeeze:
                            data_to_weight = np_squeeze(data_to_weight, axis=tuple(squeeze_out))
                        weighted_value = np_atleast_1d(np_ma_average(data_to_weight, weights=weights))
                        target[w_slc].get_value()[:] = weighted_value
                else:
                    target_to_weight = var_to_weight.get_masked_value()
                    # Sort to minimize floating point sum errors.
                    target_to_weight = target_to_weight.flatten()
                    weights = weights.flatten()
                    sindices = np.argsort(target_to_weight)
                    target_to_weight = target_to_weight[sindices]
                    weights = weights[sindices]

                    weighted_value = np.atleast_1d(np.ma.average(target_to_weight, weights=weights))
                    target = ret.parent[var_to_weight.name]
                    target.set_mask(None)
                    target.set_value(None)
                    target.set_dimensions(new_dimensions)
                    target.set_value(weighted_value)

            # Collect areas of live ranks and convert to weights.
            if vm.size > 1:
                # If there is no area information (points for example, we need to use counts).
                if ret.area.data[0].max() == 0:
                    weight_or_proxy = float(self.size)
                else:
                    weight_or_proxy = ret.area.data[0]

                if vm.rank != root:
                    vm.comm.send(weight_or_proxy, dest=root)
                else:
                    live_rank_areas = [weight_or_proxy]
                    for tner in vm.ranks:
                        if tner != vm.rank:
                            recv_area = vm.comm.recv(source=tner)
                            live_rank_areas.append(recv_area)
                    live_rank_areas = np.array(live_rank_areas)

                    rank_weights = live_rank_areas / np.max(live_rank_areas)

                for var_to_weight in filter(lambda ii: ii.name in variable_names_to_weight, list(ret.parent.values())):
                    dimensions_to_itr = [dim.name for dim in var_to_weight.dimensions if
                                         dim.name != union_dimension.name]
                    slc = {union_dimension.name: 0}
                    for idx_slc in var_to_weight.iter_dict_slices(dimensions=dimensions_to_itr):
                        idx_slc.update(slc)
                        to_weight = var_to_weight[idx_slc].get_value().flatten()[0]
                        if vm.rank == root:
                            collected_to_weight = [to_weight]
                        if not vm.rank == root:
                            vm.comm.send(to_weight, dest=root)
                        else:
                            for tner in vm.ranks:
                                if not tner == root:
                                    recv_to_weight = vm.comm.recv(source=tner)
                                    collected_to_weight.append(recv_to_weight)

                            # Sort to minimize floating point sum errors.
                            collected_to_weight = np.array(collected_to_weight)
                            sindices = np.argsort(collected_to_weight)
                            collected_to_weight = collected_to_weight[sindices]
                            rank_weights = rank_weights[sindices]

                            weighted = np.atleast_1d(np.ma.average(collected_to_weight, weights=rank_weights))
                            var_to_weight[idx_slc].get_value()[:] = weighted
        if vm.rank == root:
            return ret
        else:
            return

    def prepare(self):
        """
        Prepare the geometry variable for spatial operations by calling its coordinate system's :meth:`ocgis.variable.crs.AbstractCRS.prepare_geometry_variable`
         method. Updates the :attr:`ocgis.GeometryVariable.is_prepared` attribute to ``True``.
        """

        # TODO: This could do things like update the coordinate system, wrapping, etc.

        if not self.is_prepared:
            crs = self.crs
            if crs is not None:
                crs.prepare_geometry_variable(self)
            self.is_prepared = True

    def update_crs(self, to_crs):
        """
        Update the coordinate system of the geometry variable in-place.
        
        :param to_crs: The destination CRS for the transformation.
        :type to_crs: :class:`~ocgis.variable.crs.AbstractCRS`
        """

        super(GeometryVariable, self).update_crs(to_crs)
        members = [self.crs, to_crs]
        contains_cartesian = any([isinstance(ii, Cartesian) for ii in members])

        if contains_cartesian:
            if isinstance(to_crs, Cartesian):
                inverse = False
            else:
                inverse = True
            self.crs.transform_geometry(to_crs, self, inverse=inverse)
        elif self.crs != to_crs:
            # Be sure and project masked geometries to maintain underlying geometries.
            r_value = self.get_value().reshape(-1)
            r_loads = wkb.loads
            r_create = ogr.CreateGeometryFromWkb
            to_sr = to_crs.sr
            from_sr = self.crs.sr
            for idx, geom in enumerate(r_value.flat):
                ogr_geom = r_create(geom.wkb)
                ogr_geom.AssignSpatialReference(from_sr)
                ogr_geom.TransformTo(to_sr)
                r_value[idx] = r_loads(ogr_geom.ExportToWkb())
        # Even if coordinate systems are measured equivalent, for consistency the new crs is the destination CRS.
        self.crs = to_crs

    def iter_records(self, use_mask=True):
        if use_mask:
            to_itr = self.get_masked_value().compressed()
        else:
            to_itr = self.get_value().flat
        r_geom_class = GEOM_TYPE_MAPPING[self.geom_type]

        for idx, geom in enumerate(to_itr):
            # Convert geometry to a multi-geometry if needed.
            if not isinstance(geom, r_geom_class):
                geom = r_geom_class([geom])
            feature = {'properties': {}, 'geometry': mapping(geom)}
            yield feature

    def set_ugid(self, variable, attr_link_name=constants.AttributeName.UNIQUE_GEOMETRY_IDENTIFIER):
        """
        Same as :meth:`~ocgis.Variable.set_ugid`, except the unique identifier name has a default value.
        """

        super(GeometryVariable, self).set_ugid(variable, attr_link_name=attr_link_name)

    def set_value(self, value, **kwargs):
        if not isinstance(value, ndarray) and value is not None:
            if isinstance(value, BaseGeometry):
                itr = [value]
                shape = 1
            else:
                itr = value
                shape = len(value)
            value = np.zeros(shape, dtype=self.dtype)
            for idx, element in enumerate(itr):
                value[idx] = element
        super(GeometryVariable, self).set_value(value, **kwargs)

    def write_vector(self, *args, **kwargs):
        kwargs = kwargs.copy()
        lself = self.copy()
        if lself.parent.geom is None:
            lself.parent.set_geom(lself)
        lself.parent.set_abstraction_geom(create_ugid=True, set_ugid_as_data=True)
        kwargs[KeywordArgument.DRIVER] = DriverKey.VECTOR
        lself.parent.write(*args, **kwargs)

    def _get_extent_(self):
        raise NotImplementedError


def get_masking_slice(intersects_mask_value, target, apply_slice=True):
    """
    Collective!
    
    :param intersects_mask_value: The mask to use for creating the slice indices.
    :type intersects_mask_value: :class:`numpy.ndarray`, dtype=bool
    :param target: The target slicable object to slice.
    :param bool apply_slice: If ``True``, apply the slice.
    """
    raise_if_empty(target)

    if intersects_mask_value is None:
        local_slice = None
    else:
        if intersects_mask_value.all():
            local_slice = None
        elif not intersects_mask_value.any():
            shp = intersects_mask_value.shape
            local_slice = [(0, shp[0]), (0, shp[1])]
        else:
            _, local_slice = get_trimmed_array_by_mask(intersects_mask_value, return_adjustments=True)
            local_slice = [(l.start, l.stop) for l in local_slice]

    if local_slice is not None:
        offset_local_slice = [None] * len(local_slice)
        for idx in range(len(local_slice)):
            offset = target.dimensions[idx].bounds_local[0]
            offset_local_slice[idx] = (local_slice[idx][0] + offset, local_slice[idx][1] + offset)
    else:
        offset_local_slice = None

    gathered_offset_local_slices = vm.gather(offset_local_slice)
    if vm.rank == 0:
        gathered_offset_local_slices = [g for g in gathered_offset_local_slices if g is not None]
        if len(gathered_offset_local_slices) == 0:
            raise_empty_subset = True
        else:
            raise_empty_subset = False
            offset_array = np.array(gathered_offset_local_slices)
            global_slice = [None] * offset_array.shape[1]
            for idx in range(len(global_slice)):
                global_slice[idx] = (np.min(offset_array[:, idx, :]), np.max(offset_array[:, idx, :]))
    else:
        global_slice = None
        raise_empty_subset = None

    raise_empty_subset = vm.bcast(raise_empty_subset)
    if raise_empty_subset:
        raise EmptySubsetError
    global_slice = vm.bcast(global_slice)
    global_slice = tuple([slice(g[0], g[1]) for g in global_slice])

    intersects_mask = Variable(name='mask_gather', value=intersects_mask_value, dimensions=target.dimensions,
                               dtype=bool)

    if apply_slice:
        if vm.size_global > 1:
            ret = target.get_distributed_slice(global_slice)
            ret_mask = intersects_mask.get_distributed_slice(global_slice)
        else:
            ret = target.__getitem__(global_slice)
            ret_mask = intersects_mask.__getitem__(global_slice)
    else:
        ret = target
        ret_mask = intersects_mask

    return ret, ret_mask, global_slice


def get_geom_type(data):
    geom_type = None
    for geom in data.flat:
        try:
            geom_type = geom.geom_type
        except AttributeError:
            # Assume this is not a geometry, but an underlying masked element.
            continue
        else:
            if geom_type.startswith('Multi'):
                break
    assert geom_type is not None
    return geom_type


def get_grid_or_geom_attr(sc, attr):
    if sc.grid is None:
        ret = getattr(sc.geom, attr)
    else:
        ret = getattr(sc.grid, attr)
    return ret


def get_ccw_oriented_and_valid_shapely_polygon(geom):
    try:
        assert geom.is_valid
    except AssertionError:
        geom = geom.buffer(0)
        assert geom.is_valid

    if not geom.exterior.is_ccw:
        geom = orient(geom)

    return geom


def geometryvariable_get_mask_from_intersects(gvar, geometry, use_spatial_index=env.USE_SPATIAL_INDEX,
                                              keep_touches=False, original_mask=None):
    # Create the fill array and reference the mask. This is the output geometry value array.
    if original_mask is None:
        original_mask = gvar.get_mask(create=True)
    fill = original_mask.copy()
    fill.fill(True)
    ref_fill_mask = fill.reshape(-1)

    # Track global indices because spatial operations only occur on non-masked values.
    global_index = np.arange(original_mask.size)
    global_index = np.ma.array(global_index, mask=original_mask).compressed()
    # Select the geometry targets. If an original mask is provided, use this. It may be modified to limit the search
    # area for intersects operations. Useful for speeding up grid subsetting operations.
    geometry_target = np.ma.array(gvar.get_value(), mask=original_mask).compressed()

    if use_spatial_index:
        si = gvar.get_spatial_index(target=geometry_target)
        # Return the indices of the geometries intersecting the target geometry, and update the mask accordingly.
        for idx in si.iter_intersects(geometry, geometry_target, keep_touches=keep_touches):
            ref_fill_mask[global_index[idx]] = False
    else:
        # Prepare the polygon for faster spatial operations.
        prepared = prep(geometry)
        # We are not keeping touches at this point. Remember the mask is an inverse.
        for idx, geom in iter_array(geometry_target, return_value=True):
            bool_value = False
            if prepared.intersects(geom):
                if not keep_touches and geometry.touches(geom):
                    bool_value = True
            else:
                bool_value = True
            ref_fill_mask[global_index[idx]] = bool_value

    return fill
