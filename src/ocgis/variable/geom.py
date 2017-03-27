from abc import ABCMeta, abstractmethod
from collections import deque
from copy import copy, deepcopy
from itertools import ifilter, product

import numpy as np
from numpy.core.multiarray import ndarray
from shapely import wkb
from shapely.geometry import Point, Polygon, MultiPolygon, mapping, MultiPoint, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import cascaded_union
from shapely.prepared import prep

from ocgis import SourcedVariable, Variable
from ocgis import constants
from ocgis import env
from ocgis.base import AbstractInterfaceObject, get_dimension_names, get_variable_names
from ocgis.base import AbstractOcgisObject
from ocgis.constants import WrapAction, KeywordArguments, HeaderNames
from ocgis.environment import ogr
from ocgis.exc import EmptySubsetError
from ocgis.util.helpers import iter_array, get_none_or_slice, get_trimmed_array_by_mask, get_swap_chain
from ocgis.variable.base import AbstractContainer, get_dimension_lengths
from ocgis.variable.iterator import Iterator
from ocgis.vm.mpi import get_nonempty_ranks, get_standard_comm_state

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

GEOM_TYPE_MAPPING = {'Polygon': Polygon, 'Point': Point, 'MultiPoint': MultiPoint, 'MultiPolygon': MultiPolygon}


class AbstractSpatialObject(AbstractInterfaceObject):
    def __init__(self, *args, **kwargs):
        self._crs_name = None
        self.crs = kwargs.pop('crs', None)
        super(AbstractInterfaceObject, self).__init__(*args, **kwargs)

    @property
    def crs(self):
        if self.parent is not None and self._crs_name is not None:
            ret = self.parent.get(self._crs_name)
        else:
            ret = None
        return ret

    @crs.setter
    def crs(self, value):
        if value is None:
            if self.crs is not None:
                self.parent.pop(self._crs_name)
                self._crs_name = None
        else:
            if self.parent is None:
                self.initialize_parent()
            if self._crs_name is not None:
                self.parent.pop(self._crs_name)
            self.parent.add_variable(value, force=True)
            self._crs_name = value.name

    @property
    def wrapped_state(self):
        if self.crs is None:
            ret = None
        else:
            ret = self.crs.get_wrapped_state(self)
        return ret

    def unwrap(self):
        if not self.is_empty:
            if self.crs is None or not self.crs.is_geographic:
                raise ValueError("Only spherical coordinate systems may be wrapped/unwrapped.")
            else:
                self.crs.wrap_or_unwrap(WrapAction.UNWRAP, self)

    def wrap(self):
        if not self.is_empty:
            if self.crs is None or not self.crs.is_geographic:
                raise ValueError("Only spherical coordinate systems may be wrapped/unwrapped.")
            else:
                self.crs.wrap_or_unwrap(WrapAction.WRAP, self)


class AbstractOperationsSpatialObject(AbstractSpatialObject):
    __metaclass__ = ABCMeta

    @property
    def envelope(self):
        return box(*self.extent)

    @property
    def extent(self):
        return self._get_extent_()

    @abstractmethod
    def update_crs(self, to_crs):
        """Update coordinate system in-place."""

        if self.is_empty:
            self.crs = to_crs
            return self

        if self.crs is None:
            msg = 'The current CRS is None and cannot be updated.'
            raise ValueError(msg)

    @abstractmethod
    def _get_extent_(self):
        """
        :returns: A tuple with order (minx, miny, maxx, maxy).
        :rtype: tuple
        """

    @abstractmethod
    def get_intersects(self, *args, **kwargs):
        """Perform an intersects operations."""

    @abstractmethod
    def get_nearest(self, target, return_indices=False):
        """Get nearest element to target geometry."""

    @abstractmethod
    def get_spatial_index(self):
        """Get the spatial index."""

    @abstractmethod
    def iter_records(self, use_mask=True):
        """Generate fiona-compatible records."""


class AbstractSpatialContainer(AbstractContainer, AbstractOperationsSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        parent = kwargs.pop('parent', None)
        name = kwargs.pop('name', None)
        AbstractContainer.__init__(self, name, parent=parent)
        AbstractOperationsSpatialObject.__init__(self, crs=crs)


class AbstractSpatialVariable(SourcedVariable, AbstractOperationsSpatialObject):
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        crs = kwargs.pop('crs', None)
        SourcedVariable.__init__(self, **kwargs)
        AbstractOperationsSpatialObject.__init__(self, crs=crs)

    def deepcopy(self, eager=False):
        ret = super(AbstractSpatialVariable, self).deepcopy(eager)
        if ret.crs is not None:
            ret.crs = ret.crs.deepcopy()
        return ret

    def extract(self):
        crs = self.crs
        ret = super(AbstractSpatialVariable, self).extract()
        if crs is not None:
            self.parent.add_variable(crs)
        return ret


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
    def __init__(self, *args, **kwargs):
        self._name_ugid = None
        self._geom_type = kwargs.pop('geom_type', 'auto')

        if kwargs.get('name') is None:
            kwargs['name'] = 'geom'

        ugid = kwargs.pop(KeywordArguments.UGID, None)

        super(GeometryVariable, self).__init__(*args, **kwargs)

        if ugid is not None:
            ugid_var = Variable(name=HeaderNames.ID_SELECTION_GEOMETRY, value=[ugid], dimensions=self.dimensions)
            self.set_ugid(ugid_var)

    @property
    def area(self):
        if self.is_empty:
            fill = None
        else:
            r_value = self.masked_value
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
        # Geometry arrays are always object arrays.
        return object

    @dtype.setter
    def dtype(self, value):
        # Geometry data types are always objects. Ignore any passed value.
        pass

    @property
    def geom_type(self):
        # tdk: This needs to be optimized. Also, should this work in parallel? A resolution limit could be used. We
        # tdk:   could also state that the first geometry is always used. The point is to avoid reading all the geometries
        # tdk:   just to identify the geometry type.
        # Geometry objects may change part counts during operations. It is better to scan and update the geometry types
        # to account for these operations.
        if self._geom_type == 'auto':
            self._geom_type = get_geom_type(self.get_value())
        return self._geom_type

    @property
    def weights(self):
        area = self.area
        area.data[area.data == 0] = 1.0
        return area / area.max()

    @property
    def ugid(self):
        if self._name_ugid is None:
            return None
        else:
            return self.parent[self._name_ugid]

    def create_ugid(self, name, start=1):
        if self.is_empty:
            value = None
        else:
            value = np.arange(start, start + self.size).reshape(self.shape)
        ret = Variable(name=name, value=value, dimensions=self.dimensions, is_empty=self.is_empty)
        self.set_ugid(ret)
        return ret

    def create_ugid_global(self, name, start=1, comm=None):
        """Collective!"""
        comm, rank, size = get_standard_comm_state(comm=comm)

        ner = get_nonempty_ranks(self)
        sizes = comm.gather(self.size)
        rank_start = None
        if rank == 0:
            start = start
            for n in ner:
                if n == 0:
                    rank_start = start
                else:
                    comm.send(start, dest=n)
                start += sizes[n]
        elif not self.is_empty:
            rank_start = comm.recv(source=0)
        else:
            pass

        return self.create_ugid(name, start=rank_start)

    def get_buffer(self, *args, **kwargs):
        """
        Return a shallow copy of the geometry variable with geometries buffered.
        """

        # Handle empty objects.
        if self.is_empty:
            return self

        # New geometry type for the buffered object.
        geom_type = kwargs.pop('geom_type', 'auto')

        ret = self.copy()
        new_value = np.empty_like(ret.value, dtype=object)
        to_buffer = self.value
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
        :param bool return_slice: (``='False'``) If ``True``, return the _global_ slice that will guarantee no masked
         elements outside the subset geometry.
        :param comm: (``=None``) If ``None``, use the default MPI communicator.
        :return:
        :raises: EmptySubsetError
        """
        return_slice = kwargs.pop(KeywordArguments.RETURN_SLICE, False)
        cascade = kwargs.pop(KeywordArguments.CASCADE, True)
        comm = kwargs.pop(KeywordArguments.COMM, None)
        comm, rank, size = get_standard_comm_state(comm=comm)

        ret = self.copy()
        intersects_mask_value = ret.get_mask_from_intersects(*args, **kwargs)
        ret, ret_mask, ret_slice = get_masking_slice(intersects_mask_value, ret, comm=comm)

        if not ret.is_empty:
            ret.set_mask(ret_mask.value, cascade=cascade, update=True)
        else:
            for var in ret.parent.values():
                assert var.is_empty

        # tdk: need to implement fancy index-based slicing for the one-dimensional unstructured case
        # if self.ndim == 1:
        #     # For one-dimensional data, assume it is unstructured and compress the returned data.
        #     adjust = np.where(np.invert(ret.get_mask()))
        #     ret_slc = adjust

        if return_slice:
            ret = (ret, ret_slice)

        return ret

    def get_intersection(self, *args, **kwargs):
        """
        Collective!

        .. note:: Accepts all parameters to :meth:`~ocgis.new_interface.geom.GeometryVariable.get_intersects`. Same
         return types.

        Additional arguments and/or keyword arguments are:

        :param str calendar: (``='standard'``)

        :param bool inplace: (``=False``) If ``False``, deep copy the geometry array on the output before executing an
         intersection. If ``True``, modify the geometries in-place.
        :param bool intersects_check: (``=True``) If ``True``, first perform an intersects operation to limit the
         geometries tests for intersection. If ``False``, perform the intersection as is.
        """

        inplace = kwargs.pop(KeywordArguments.INPLACE, False)
        intersects_check = kwargs.pop(KeywordArguments.INTERSECTS_CHECK, True)
        return_slice = kwargs.get(KeywordArguments.RETURN_SLICE, False)
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
                obj.set_value(deepcopy(obj.value))
            obj_value = obj.get_masked_value()
            for idx, geom in iter_array(obj_value, return_value=True):
                obj_value.data[idx] = geom.intersection(subset_geometry)
        return ret

    def get_iter(self, *args, **kwargs):
        should_add = kwargs.pop(KeywordArguments.ADD_GEOM_UID, False)

        if should_add and self.ugid is not None:
            followers = [self.ugid]
        else:
            followers = []

        return Iterator(self, followers=followers, **kwargs)

    def get_mask_from_intersects(self, geometry_or_bounds, use_spatial_index=True, keep_touches=False,
                                 original_mask=None):
        # Transform bounds sequence to a geometry.
        if not isinstance(geometry_or_bounds, BaseGeometry):
            geometry_or_bounds = box(*geometry_or_bounds)

        # Handle empty variables.
        if self.is_empty:
            ret = None
        else:
            ret = geometryvariable_get_mask_from_intersects(self, geometry_or_bounds,
                                                            use_spatial_index=use_spatial_index,
                                                            keep_touches=keep_touches,
                                                            original_mask=original_mask)
        return ret

    def get_nearest(self, target, return_indices=False):
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
        Return a spatial index for the geometry variable.
        :param target: If this is a boolean array, use this as the add target. Otherwise, use the compressed masked
         values.
        :return:
        """
        # "rtree" is an optional dependency.
        from ocgis.spatial.index import SpatialIndex
        # Fill the spatial index with unmasked values only.
        si = SpatialIndex()
        # Use compressed masked values if target is not available.
        if target is None:
            target = self.masked_value.compressed()
        # Add the geometries to the index.
        r_add = si.add
        for idx, geom in iter_array(target, return_value=True):
            r_add(idx[0], geom)

        return si

    def get_unioned(self, dimensions=None, union_dimension=None, spatial_average=None, comm=None):
        """
        Unions _unmasked_ geometry objects. Collective across communicator.
        """
        # tdk: optimize

        comm, rank, size = get_standard_comm_state(comm=comm)
        ner = get_nonempty_ranks(self)
        if rank == 0:
            root = ner[0]
        else:
            root = None
        root = comm.bcast(root)
        ner = comm.bcast(ner)

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
            union_dimension = Dimension(constants.DimensionNames.UNIONED_GEOMETRY, 1)
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

        if not self.is_empty:
            # Destination indices in the return variable are filled with non-masked, unioned geometries.
            for dst_indices in product(*[range(dl) for dl in get_dimension_lengths(new_dimensions)]):
                dst_slc = {new_dimensions[ii].name: dst_indices[ii] for ii in range(len(new_dimensions))}

                # Select the geometries to union skipping any masked geometries.
                to_union = deque()
                for indices in product(*[range(dl) for dl in dimension_lengths]):
                    dslc = {dimension_names[ii]: indices[ii] for ii in range(len(dimension_names))}
                    sub = self[dslc]
                    sub_mask = sub.get_mask()
                    if sub_mask is None:
                        to_union.append(sub.value.flatten()[0])
                    else:
                        if not sub_mask.flatten()[0]:
                            to_union.append(sub.value.flatten()[0])

                # Execute the union operation.
                processed_to_union = deque()
                for geom in to_union:
                    if isinstance(geom, MultiPolygon) or isinstance(geom, MultiPoint):
                        for element in geom:
                            processed_to_union.append(element)
                    else:
                        processed_to_union.append(geom)
                unioned = cascaded_union(processed_to_union)

                # Fill the return geometry variable value with the unioned geometry.
                to_fill = ret[dst_slc].get_value()
                to_fill[0] = unioned

        # Spatial average shared dimensions.
        if spatial_average is not None and not self.is_empty:
            # Get source data to weight.
            for var_to_weight in ifilter(lambda ii: ii.name in variable_names_to_weight, self.parent.values()):
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

                # Path is there are iteration dimensions. Checks for axes ordering in addition.
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
                    for nonweighted_indices in product(*[range(ri) for ri in range_to_itr]):
                        w_slc = {names_to_itr[ii]: nonweighted_indices[ii] for ii in range(len_names_to_itr)}
                        for nsa in names_to_slice_all:
                            w_slc[nsa] = slice_none
                        data_to_weight = var_to_weight[w_slc].masked_value
                        if should_squeeze:
                            data_to_weight = np_squeeze(data_to_weight, axis=tuple(squeeze_out))
                        weighted_value = np_atleast_1d(np_ma_average(data_to_weight, weights=weights))
                        target[w_slc].value[:] = weighted_value
                else:
                    weighted_value = np.atleast_1d(np.ma.average(var_to_weight.masked_value, weights=weights))
                    target = ret.parent[var_to_weight.name]
                    target.set_mask(None)
                    target.set_value(None)
                    target.set_dimensions(new_dimensions)
                    target.set_value(weighted_value)

            # Collect areas of live ranks and convert to weights.
            if size > 1:
                root = ner[0]
                if rank == root:
                    live_rank_areas = [ret.area.data[0]]
                if rank in ner and rank != root:
                    comm.send(ret.area.data[0], dest=root)
                else:
                    for tner in ner:
                        if tner != rank:
                            recv_area = comm.recv(source=tner)
                            live_rank_areas.append(recv_area)
                    live_rank_areas = np.array(live_rank_areas)
                    rank_weights = live_rank_areas / np.max(live_rank_areas)

                for var_to_weight in ifilter(lambda ii: ii.name in variable_names_to_weight, ret.parent.values()):
                    dimensions_to_itr = [dim.name for dim in var_to_weight.dimensions if
                                         dim.name != union_dimension.name]
                    slc = {union_dimension.name: 0}
                    for idx_slc in var_to_weight.iter_dict_slices(dimensions=dimensions_to_itr):
                        idx_slc.update(slc)
                        to_weight = var_to_weight[idx_slc].value.flatten()[0]

                        if rank == root:
                            collected_to_weight = [to_weight]
                        if rank in ner and rank != root:
                            comm.send(to_weight, dest=root)
                        else:
                            for tner in ner:
                                if tner != root:
                                    recv_to_weight = comm.recv(source=tner)
                                    collected_to_weight.append(recv_to_weight)
                            weighted = np.atleast_1d(np.ma.average(collected_to_weight, weights=rank_weights))
                            var_to_weight[idx_slc].value[:] = weighted

        if rank == root:
            return ret
        else:
            return None

    def update_crs(self, to_crs):
        super(GeometryVariable, self).update_crs(to_crs)

        if self.is_empty:
            return

        # Be sure and project masked geometries to maintain underlying geometries.
        r_value = self.value.reshape(-1)
        r_loads = wkb.loads
        r_create = ogr.CreateGeometryFromWkb
        to_sr = to_crs.sr
        from_sr = self.crs.sr
        for idx, geom in enumerate(r_value.flat):
            ogr_geom = r_create(geom.wkb)
            ogr_geom.AssignSpatialReference(from_sr)
            ogr_geom.TransformTo(to_sr)
            r_value[idx] = r_loads(ogr_geom.ExportToWkb())
        self.crs = to_crs

    def iter_records(self, use_mask=True):
        if use_mask:
            to_itr = self.masked_value.compressed()
        else:
            to_itr = self.value.flat
        r_geom_class = GEOM_TYPE_MAPPING[self.geom_type]

        for idx, geom in enumerate(to_itr):
            # Convert geometry to a multi-geometry if needed.
            if not isinstance(geom, r_geom_class):
                geom = r_geom_class([geom])
            feature = {'properties': {}, 'geometry': mapping(geom)}
            yield feature

    def set_ugid(self, variable):
        if variable is None:
            self._name_ugid = None
            self.attrs.pop(constants.AttributeNames.UNIQUE_GEOMETRY_IDENTIFIER, None)
        else:
            self.parent.add_variable(variable, force=True)
            self._name_ugid = variable.name
            self.attrs[constants.AttributeNames.UNIQUE_GEOMETRY_IDENTIFIER] = variable.name

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
        from ocgis.collection.field import OcgField
        from ocgis.driver.vector import DriverVector
        field = OcgField(geom=self, crs=self.crs)
        kwargs[KeywordArguments.DRIVER] = DriverVector
        field.write(*args, **kwargs)

    def _get_extent_(self):
        raise NotImplementedError


def get_masking_slice(intersects_mask_value, target, apply_slice=True, comm=None):
    """Collective!"""
    comm, rank, size = get_standard_comm_state(comm=comm)

    if intersects_mask_value is None or target.is_empty:
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

    gathered_offset_local_slices = comm.gather(offset_local_slice)
    if rank == 0:
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
    raise_empty_subset = comm.bcast(raise_empty_subset)
    if raise_empty_subset:
        raise EmptySubsetError
    global_slice = comm.bcast(global_slice)
    global_slice = tuple([slice(g[0], g[1]) for g in global_slice])

    if local_slice is None:
        intersects_mask_is_empty = True
    else:
        intersects_mask_is_empty = False

    intersects_mask = Variable(name='mask_gather', value=intersects_mask_value, dimensions=target.dimensions,
                               dtype=bool, is_empty=intersects_mask_is_empty)

    if apply_slice:
        if size > 1:
            ret = target.get_distributed_slice(global_slice, comm=comm)
            ret_mask = intersects_mask.get_distributed_slice(global_slice, comm=comm)
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


def get_spatial_operation(sc, name, args, kwargs):
    """
    :param sc: A spatial containter.
    :type sc: :class:`ocgis.new_interface.geom.SpatialContainer'
    :param str name: Name of the spatial operation.
    :param tuple args: Arguments to the spatial operation.
    :param dict kwargs: Keyword arguments to the spatial operation.
    :returns: Performs the spatial operation on the input spatial container.
    :rtype: :class:`ocgis.new_interface.geom.SpatialContainer'
    """
    ret = copy(sc)

    # Always return indices so the other geometry can be sliced if needed.
    kwargs = kwargs.copy()
    original_return_indices = kwargs.get('return_indices', False)
    kwargs['return_indices'] = True

    # Subset the optimal geometry.
    geom = ret.geom
    operation = getattr(geom, name)
    geom_subset, slc = operation(*args, **kwargs)
    # Synchronize the underlying grid.
    ret.grid = geom_subset.grid
    # Update the other geometry by slicing given the underlying subset. Only slice if it is loaded.
    if isinstance(geom, PolygonArray):
        ret.polygon = geom_subset
        ret.point = get_none_or_slice(ret._point, slc)
    else:
        ret.point = geom_subset
        ret.polygon = get_none_or_slice(ret._polygon, slc)

    if original_return_indices:
        ret = (ret, slc)
    else:
        ret = ret

    return ret


def geometryvariable_get_mask_from_intersects(gvar, geometry, use_spatial_index=True, keep_touches=False,
                                              original_mask=None):
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
    geometry_target = np.ma.array(gvar.value, mask=original_mask).compressed()

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
