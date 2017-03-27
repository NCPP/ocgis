import itertools

import numpy as np
from pyproj import Proj, transform
from shapely.geometry import Polygon, Point, box
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from ocgis import VariableCollection, Variable
from ocgis import constants
from ocgis.base import get_dimension_names
from ocgis.constants import WrappedState, KeywordArguments, VariableNames
from ocgis.environment import ogr
from ocgis.exc import GridDeficientError, EmptySubsetError, AllElementsMaskedError
from ocgis.util.helpers import get_formatted_slice
from ocgis.variable.base import get_dslice, get_dimension_lengths
from ocgis.variable.crs import CFRotatedPole
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable, AbstractSpatialContainer, get_masking_slice, GeometryProcessor
from ocgis.vm.mpi import MPI_COMM, get_standard_comm_state, \
    get_nonempty_ranks, MPI_RANK, MPI_SIZE

CreateGeometryFromWkb, Geometry, wkbGeometryCollection, wkbPoint = ogr.CreateGeometryFromWkb, ogr.Geometry, \
                                                                   ogr.wkbGeometryCollection, ogr.wkbPoint

_NAMES_2D = ['ocgis_yc', 'ocgis_xc']


class GridGeometryProcessor(GeometryProcessor):
    def __init__(self, grid, subset_geometry, hint_mask, keep_touches=False, use_bounds=True):
        if hint_mask is not None:
            assert hint_mask.ndim == 2
            assert hint_mask.dtype == np.bool

        self.use_bounds = use_bounds
        self.grid = grid
        self.hint_mask = hint_mask
        geometry_iterable = self.get_geometry_iterable()
        super(GridGeometryProcessor, self).__init__(geometry_iterable, subset_geometry, keep_touches=keep_touches)

    def get_geometry_iterable(self):
        grid = self.grid
        hint_mask = self.hint_mask
        is_vectorized = grid.is_vectorized
        if self.use_bounds:
            abstraction = grid.abstraction
        else:
            abstraction = 'point'

        if abstraction == 'point':
            x_data = grid.x.value
            y_data = grid.y.value
            for idx_row, idx_col in itertools.product(*[range(ii) for ii in grid.shape]):
                if hint_mask is not None and hint_mask[idx_row, idx_col]:
                    yld = None
                else:
                    if is_vectorized:
                        y = y_data[idx_row]
                        x = x_data[idx_col]
                    else:
                        y = y_data[idx_row, idx_col]
                        x = x_data[idx_row, idx_col]
                    yld = Point(x, y)
                yield (idx_row, idx_col), yld
        elif abstraction == 'polygon':
            if grid.has_bounds:
                # We want geometries for everything even if masked.
                x_bounds = grid.x.bounds.value
                y_bounds = grid.y.bounds.value
                range_row = range(grid.shape[0])
                range_col = range(grid.shape[1])
                if is_vectorized:
                    for row, col in itertools.product(range_row, range_col):
                        if hint_mask is not None and hint_mask[row, col]:
                            polygon = None
                        else:
                            min_x, max_x = np.min(x_bounds[col, :]), np.max(x_bounds[col, :])
                            min_y, max_y = np.min(y_bounds[row, :]), np.max(y_bounds[row, :])
                            polygon = box(min_x, min_y, max_x, max_y)
                        yield (row, col), polygon
                else:
                    # tdk: we should be able to avoid the creation of this corners array
                    corners = np.vstack((y_bounds, x_bounds))
                    corners = corners.reshape([2] + list(x_bounds.shape))
                    for row, col in itertools.product(range_row, range_col):
                        if hint_mask is not None and hint_mask[row, col]:
                            polygon = None
                        else:
                            current_corner = corners[:, row, col]
                            coords = np.hstack((current_corner[1, :].reshape(-1, 1),
                                                current_corner[0, :].reshape(-1, 1)))
                            polygon = Polygon(coords)
                        yield (row, col), polygon
            else:
                msg = 'A grid must have bounds/corners to construct polygons. Consider using "set_extrapolated_bounds".'
                raise GridDeficientError(msg)
        else:
            raise NotImplementedError(abstraction)


class GridXY(AbstractSpatialContainer):
    ndim = 2

    def __init__(self, x, y, abstraction='auto', crs=None, parent=None, mask=None):
        if x.dimensions is None or y.dimensions is None:
            raise ValueError('Grid variables must have dimensions.')
        if abstraction is None:
            raise ValueError('"abstraction" may not be None.')

        self._abstraction = None

        self.abstraction = abstraction

        x.attrs['axis'] = 'X'
        y.attrs['axis'] = 'Y'

        self._x_name = x.name
        self._y_name = y.name

        if mask is None:
            self._mask_name = VariableNames.SPATIAL_MASK
        else:
            self._mask_name = mask.name
            self.parent.add_variable(mask)

        self._point_name = VariableNames.GEOMETRY_POINT
        self._polygon_name = VariableNames.GEOMETRY_POLYGON

        new_variables = [x, y]
        if parent is None:
            parent = VariableCollection(variables=new_variables)
        else:
            for var in new_variables:
                parent.add_variable(var, force=True)

        super(GridXY, self).__init__(crs=crs, parent=parent)

    def __getitem__(self, slc):
        """
        :param slc: The slice sequence with indices corresponding to:

         0 --> y-dimension
         1 --> x-dimension

        :type slc: sequence of slice-compatible arguments
        :returns: Sliced grid.
        :rtype: :class:`ocgis.new_interface.grid.GridXY`
        """

        if not isinstance(slc, dict):
            slc = get_dslice(self.dimensions, slc)
        ret = self.copy()
        new_parent = ret.parent[slc]
        ret.parent = new_parent
        return ret

    def __setitem__(self, slc, grid):
        slc = get_formatted_slice(slc, self.ndim)

        if self.is_vectorized:
            y_slc, x_slc = slc
            self.x[x_slc] = grid.x
            self.y[y_slc] = grid.y
        else:
            self.x[slc] = grid.x
            self.y[slc] = grid.y

        if grid.mask_variable is not None:
            new_mask = grid.get_mask(create=True)
            current_mask = self.get_mask(create=True)
            current_mask[slc] = new_mask
            self.set_mask(current_mask)

    def get_member_variables(self, include_bounds=True):
        targets = [self._x_name, self._y_name, self._point_name, self._polygon_name, self._mask_name]
        ret = []
        for target in targets:
            try:
                var = self.parent[target]
            except KeyError:
                pass
            else:
                ret.append(var)
                if include_bounds and var.has_bounds:
                    ret.append(var.bounds)
        return ret

    @property
    def dtype(self):
        return self.archetype.dtype

    @property
    def extent_global(self):
        return get_extent_global(self)

    @property
    def has_allocated_point(self):
        if self._point_name in self.parent:
            return True
        else:
            return False

    @property
    def has_allocated_polygon(self):
        if self._polygon_name in self.parent:
            return True
        else:
            return False

    @property
    def has_allocated_abstraction_geometry(self):
        if self.abstraction == 'point':
            return self.has_allocated_point
        elif self.abstraction == 'polygon':
            return self.has_allocated_polygon
        else:
            raise NotImplementedError(self.abstraction)

    @property
    def is_vectorized(self):
        ndim = self.archetype.ndim
        if ndim == 1:
            ret = True
        else:
            ret = False
        return ret

    @property
    def dimensions(self):
        ret = self.archetype.dimensions
        if len(ret) == 1:
            ret = (self.parent[self._y_name].dimensions[0], self.parent[self._x_name].dimensions[0])
        return ret

    # tdk: REMOVE
    @property
    def dist(self):
        raise NotImplementedError

    @property
    def has_bounds(self):
        return self.archetype.has_bounds

    @property
    def is_empty(self):
        x = self.parent[self._x_name]
        y = self.parent[self._y_name]
        if x.is_empty or y.is_empty:
            ret = True
        else:
            ret = False
        return ret

    @property
    def mask_variable(self):
        return self.parent.get(self._mask_name)

    def get_point(self, value=None, mask=None):
        return get_geometry_variable(self, value=value, mask=mask, use_bounds=False)

    def get_polygon(self, value=None, mask=None):
        return get_geometry_variable(self, value=value, mask=mask, use_bounds=True)

    @property
    def x(self):
        ret = self.parent[self._x_name]
        return ret

    @x.setter
    def x(self, value):
        self.parent[self._x_name] = value

    @property
    def y(self):
        ret = self.parent[self._y_name]
        return ret

    @y.setter
    def y(self, value):
        self.parent[self._y_name] = value

    @property
    def resolution(self):
        y_value = self.y.value
        x_value = self.x.value
        resolution_limit = constants.RESOLUTION_LIMIT
        if self.is_vectorized:
            targets = [np.abs(np.diff(np.abs(y_value[0:resolution_limit]))),
                       np.abs(np.diff(np.abs(x_value[0:resolution_limit])))]
        else:
            targets = [np.abs(np.diff(np.abs(y_value[0:resolution_limit, :]), axis=0)),
                       np.abs(np.diff(np.abs(x_value[:, 0:resolution_limit]), axis=1))]
        to_mean = [np.mean(t) for t in targets]
        ret = np.mean(to_mean)
        return ret

    @property
    def shape(self):
        return get_dimension_lengths(self.dimensions)

    @property
    def shape_global(self):
        """Collective!"""

        ner = get_nonempty_ranks(self)
        ner = MPI_COMM.bcast(ner)

        if self.is_empty:
            maxd = None
        else:
            maxd = [max(d.bounds_global) for d in self.dimensions]
        shapes = MPI_COMM.gather(maxd)
        if MPI_RANK == 0:
            shapes = [shapes[ii] for ii in ner]
            shape_global = tuple(np.max(shapes, axis=0))
        else:
            shape_global = None
        shape_global = MPI_COMM.bcast(shape_global)

        return shape_global

    def get_value_stacked(self):
        y = self.y.get_value()
        x = self.x.get_value()

        if self.is_vectorized:
            x, y = np.meshgrid(x, y)

        fill = np.zeros([2] + list(y.shape))
        fill[0, :, :] = y
        fill[1, :, :] = x
        return fill

    @property
    def archetype(self):
        return self.parent[self._y_name]

    def expand(self):
        expand_grid(self)

    def get_mask(self, *args, **kwargs):
        create = kwargs.get(KeywordArguments.CREATE, False)
        mask_variable = self.mask_variable
        ret = None
        if mask_variable is None:
            if create:
                mask_variable = create_grid_mask_variable(self._mask_name, None, self.dimensions)
                self.parent.add_variable(mask_variable)
        if mask_variable is not None:
            ret = mask_variable.get_mask(*args, **kwargs)
            if mask_variable.attrs.get('ocgis_role') != 'spatial_mask':
                msg = 'Mask variable "{}" must have an "ocgis_role" attribute with a value of "spatial_mask".'.format(
                    ret.name)
                raise ValueError(msg)
        return ret

    def is_abstraction_available(self, abstraction):
        if abstraction == 'auto':
            ret = True
        elif abstraction == 'point':
            ret = True
        elif abstraction == 'polygon':
            if self.archetype.has_bounds:
                ret = True
            else:
                ret = False
        else:
            raise NotImplementedError(abstraction)
        return ret

    def set_mask(self, value, cascade=False):
        mask_variable = self.mask_variable
        if mask_variable is None:
            mask_variable = create_grid_mask_variable(self._mask_name, value, self.dimensions)
            self.parent.add_variable(mask_variable)
        else:
            mask_variable.set_mask(value)

        if cascade:
            grid_set_mask_cascade(self)

    def copy(self):
        ret = super(GridXY, self).copy()
        ret.parent = ret.parent.copy()
        return ret

    def get_distributed_slice(self, slc, **kwargs):
        """Collective!"""
        if isinstance(slc, dict):
            y_slc = slc[self.dimensions[0].name]
            x_slc = slc[self.dimensions[1].name]
            slc = [y_slc, x_slc]

        ret = self.copy()
        if self.is_vectorized:
            dummy_var = Variable(name='__ocgis_dummy_var__', dimensions=ret.dimensions)
            ret.parent.add_variable(dummy_var)
            ret.parent = dummy_var.get_distributed_slice(slc, **kwargs).parent
            ret.parent.pop(dummy_var.name)
        else:
            ret.parent = self.archetype.get_distributed_slice(slc, **kwargs).parent
        return ret

    def get_intersects(self, *args, **kwargs):
        args = list(args)
        args.insert(0, 'intersects')
        return self.get_spatial_operation(*args, **kwargs)

    def get_intersection(self, *args, **kwargs):
        args = list(args)
        args.insert(0, 'intersection')
        return self.get_spatial_operation(*args, **kwargs)

    def get_spatial_operation(self, spatial_op, subset_geom, return_slice=False, use_bounds='auto', original_mask=None,
                              keep_touches='auto', cascade=True, optimized_bbox_subset=False, apply_slice=True,
                              comm=None):

        comm, rank, size = get_standard_comm_state(comm)
        original_subset_target = subset_geom

        if use_bounds is True and self.abstraction == 'point':
            msg = '"use_bounds" is True and grid abstraction is "point". Only a polygon abstraction may use bounds ' \
                  'during a spatial subset operation.'
            raise ValueError(msg)

        if not isinstance(original_subset_target, BaseGeometry):
            msg = 'Only Shapely geometries allowed for subsetting. Subset type is "{}".'.format(
                type(original_subset_target))
            raise ValueError(msg)

        if use_bounds == 'auto':
            if self.abstraction == 'polygon':
                use_bounds = True
            else:
                use_bounds = False

        if spatial_op == 'intersection':
            perform_intersection = True
        else:
            perform_intersection = False

        if keep_touches == 'auto':
            if self.abstraction == 'point' or not use_bounds:
                keep_touches = True
            else:
                keep_touches = False

        buffer_value = None

        if original_mask is None:
            if not self.is_empty:
                if not optimized_bbox_subset:
                    buffer_value = self.resolution * 1.25

                if isinstance(original_subset_target, BaseMultipartGeometry):
                    geom_itr = original_subset_target
                else:
                    geom_itr = [original_subset_target]

                for ctr, geom in enumerate(geom_itr):
                    if not optimized_bbox_subset:
                        geom = geom.buffer(buffer_value).envelope
                    single_hint_mask = get_hint_mask_from_geometry_bounds(self, geom.bounds, invert=False)

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

        ret = self.copy()
        if optimized_bbox_subset:
            sliced_grid, _, the_slice = get_masking_slice(original_mask, ret, apply_slice=apply_slice, comm=comm)
        else:
            fill_mask = None
            geometry_fill = None
            if not self.is_empty:
                # If everything is masked, there is no reason to load the grid geometries.
                if not original_mask.all():
                    fill_mask = original_mask
                    if perform_intersection:
                        geometry_fill = np.zeros(fill_mask.shape, dtype=object)
                    if size > 1:
                        new_intersects_target = original_subset_target.intersection(box(*self.extent).buffer(1e-6))
                    else:
                        new_intersects_target = original_subset_target
                    gp = GridGeometryProcessor(self, new_intersects_target, original_mask, keep_touches=keep_touches,
                                               use_bounds=use_bounds)
                    for idx, intersects_logical, current_geometry in gp.iter_intersects():
                        fill_mask[idx] = not intersects_logical
                        if perform_intersection and intersects_logical:
                            geometry_fill[idx] = current_geometry.intersection(original_subset_target)
                            # fill_mask = fill_mask.reshape(*original_mask.shape)
            if perform_intersection:
                if geometry_fill is None:
                    if use_bounds:
                        name = self._polygon_name
                    else:
                        name = self._point_name
                    geometry_variable = GeometryVariable(is_empty=True, name=name)
                else:
                    if use_bounds:
                        geometry_variable = ret.get_polygon(value=geometry_fill, mask=fill_mask)
                    else:
                        geometry_variable = ret.get_point(value=geometry_fill, mask=fill_mask)
                ret.parent.add_variable(geometry_variable, force=True)
            sliced_grid, sliced_mask, the_slice = get_masking_slice(fill_mask, ret, apply_slice=apply_slice, comm=comm)
            sliced_grid.set_mask(sliced_mask.get_value(), cascade=cascade)

        if perform_intersection:
            obj_to_ret = sliced_grid.parent[geometry_variable.name]
        else:
            obj_to_ret = sliced_grid

        if return_slice:
            ret = (obj_to_ret, the_slice)
        else:
            ret = obj_to_ret

        return ret

    def set_extrapolated_bounds(self, name_x_variable, name_y_variable, name_dimension):
        """
        Extrapolate corners from grid centroids.
        """
        self.x.set_extrapolated_bounds(name_x_variable, name_dimension)
        self.y.set_extrapolated_bounds(name_y_variable, name_dimension)
        self.parent = self.y.parent

    def update_crs(self, to_crs):
        """
        Update the coordinate system in place.

        :param to_crs: The destination coordinate system.
        :type to_crs: :class:`ocgis.interface.base.crs.CoordinateReferenceSystem`
        """
        super(GridXY, self).update_crs(to_crs)

        # The superclass updates the coordinate system.
        if self.is_empty:
            return

        if isinstance(self.crs, CFRotatedPole):
            self.crs.update_with_rotated_pole_transformation(self, inverse=False)
        elif isinstance(to_crs, CFRotatedPole):
            to_crs.update_with_rotated_pole_transformation(self, inverse=True)
        else:
            # Rotated pole transforms can maintain grid vectorization. Other coordinate transforms require the grid
            # be expanded.
            expand_grid(self)

            src_proj4 = self.crs.proj4
            dst_proj4 = to_crs.proj4

            src_proj4 = Proj(src_proj4)
            dst_proj4 = Proj(dst_proj4)

            y = self.y
            x = self.x

            value_row = self.y.value.reshape(-1)
            value_col = self.x.value.reshape(-1)

            tvalue_col, tvalue_row = transform(src_proj4, dst_proj4, value_col, value_row)

            self.x.set_value(tvalue_col.reshape(self.shape))
            self.y.set_value(tvalue_row.reshape(self.shape))

            if self.has_bounds:
                corner_row = y.bounds.value.reshape(-1)
                corner_col = x.bounds.value.reshape(-1)
                tvalue_col, tvalue_row = transform(src_proj4, dst_proj4, corner_col, corner_row)
                y.bounds.set_value(tvalue_row.reshape(y.bounds.shape))
                x.bounds.set_value(tvalue_col.reshape(x.bounds.shape))

        self.crs = to_crs

        # Regenerate geometries.
        self.point = None
        self.polygon = None

    def _get_extent_(self):
        if self.is_empty:
            return None

        # tdk: test, doc
        if not self.is_vectorized:
            if self.has_bounds:
                x_bounds = self.x.bounds.value
                y_bounds = self.y.bounds.value
                minx = x_bounds.min()
                miny = y_bounds.min()
                maxx = x_bounds.max()
                maxy = y_bounds.max()
            else:
                x_value = self.x.value
                y_value = self.y.value
                minx = x_value.min()
                miny = y_value.min()
                maxx = x_value.max()
                maxy = y_value.max()
        else:
            row = self.y
            col = self.x
            if not self.has_bounds:
                minx = col.value.min()
                miny = row.value.min()
                maxx = col.value.max()
                maxy = row.value.max()
            else:
                minx = col.bounds.value.min()
                miny = row.bounds.value.min()
                maxx = col.bounds.value.max()
                maxy = row.bounds.value.max()
        return minx, miny, maxx, maxy

    @property
    def abstraction(self):
        if self._abstraction == 'auto':
            if self.has_bounds:
                ret = 'polygon'
            else:
                ret = 'point'
        else:
            ret = self._abstraction
        return ret

    @abstraction.setter
    def abstraction(self, abstraction):
        self._abstraction = abstraction

    def get_abstraction_geometry(self, **kwargs):
        if self.abstraction == 'point':
            ret = self.get_point(**kwargs)
        elif self.abstraction == 'polygon':
            ret = self.get_polygon(**kwargs)
        else:
            raise NotImplementedError(self.abstraction)
        return ret

    def get_nearest(self, *args, **kwargs):
        ret = self.copy()
        _, slc = self.get_abstraction_geometry().get_nearest(*args, **kwargs)
        ret = ret.__getitem__(slc)
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
                 'Extent = {0}'.format(self.extent),
                 'Resolution = {0}'.format(self.resolution)]

        return lines

    def get_spatial_index(self, *args, **kwargs):
        return self.abstraction_geometry.get_spatial_index(*args, **kwargs)

    def iter_records(self, *args, **kwargs):
        return self.abstraction_geometry.iter_records(self, *args, **kwargs)

    def remove_bounds(self):
        self.x.set_bounds(None)
        self.y.set_bounds(None)

    def reorder(self):
        if self.wrapped_state != WrappedState.WRAPPED:
            raise ValueError('Only wrapped coordinates may be reordered.')

        if self.is_empty:
            return self

        reorder_dimension = self.dimensions[1].name
        varying_dimension = self.dimensions[0].name

        if self.dimensions[1].dist and MPI_SIZE > 1:
            raise ValueError('The reorder dimension may not be distributed.')

        # Reorder indices identify where the index translation occurs.
        wrapped = self.x.value

        if self.is_vectorized:
            wrapped = wrapped.reshape(1, -1)

        shift_indices = np.zeros(self.shape[0], dtype=int)
        for row_index in range(wrapped.shape[0]):
            the_split_index = None
            for ctr, element in enumerate(wrapped[row_index].flat):
                if element < 0:
                    the_split_index = ctr
                    break
            shift_indices[row_index] = the_split_index

        if self.is_vectorized:
            shift_indices[:] = shift_indices[0]
            reorder_array(shift_indices, self.x.value.reshape(1, -1), get_dimension_names(self.dimensions),
                          reorder_dimension, varying_dimension)

        # Reorder all arrays that have the reorder and varying dimension.
        for var in self.parent.values():
            arr_dimension_names = get_dimension_names(var.dimensions)
            if reorder_dimension in arr_dimension_names and varying_dimension in arr_dimension_names:
                reorder_array(shift_indices, var.value, arr_dimension_names, reorder_dimension, varying_dimension)
                if var.has_masked_values:
                    mask = var.get_mask()
                    if mask is not None:
                        reorder_array(shift_indices, mask, arr_dimension_names, reorder_dimension, varying_dimension)
                        var.set_mask(mask)

    def write_fiona(self, *args, **kwargs):
        return self.abstraction_geometry.write_fiona(*args, **kwargs)

    def write(self, *args, **kwargs):
        from ocgis.driver.nc import DriverNetcdfCF
        from ocgis.collection.field import OcgField

        kwargs['driver'] = kwargs.pop('driver', DriverNetcdfCF)
        field_to_write = OcgField(grid=self)
        field_to_write.write(*args, **kwargs)


def create_grid_mask_variable(name, mask_value, dimensions):
    mask_variable = Variable(name, mask=mask_value, dtype=np.dtype('i1'), dimensions=dimensions,
                             fill_value=np.array([1], dtype=np.dtype('i1'))[0],
                             attrs={'ocgis_role': 'spatial_mask',
                                    'description': '1=True (is masked); 0=False (not masked)'})
    return mask_variable


def update_crs_with_geometry_collection(src_sr, to_sr, value_row, value_col):
    """
    Update coordinate vectors in place to match the destination coordinate system.

    :param src_sr: The source coordinate system.
    :type src_sr: :class:`osgeo.osr.SpatialReference`
    :param to_sr: The destination coordinate system.
    :type to_sr: :class:`osgeo.osr.SpatialReference`
    :param value_row: Vector of row or Y values.
    :type value_row: :class:`numpy.ndarray`
    :param value_col: Vector of column or X values.
    :type value_col: :class:`numpy.ndarray`
    """

    geomcol = Geometry(wkbGeometryCollection)
    for ii in range(value_row.shape[0]):
        point = Geometry(wkbPoint)
        point.AddPoint(value_col[ii], value_row[ii])
        geomcol.AddGeometry(point)
    geomcol.AssignSpatialReference(src_sr)
    geomcol.TransformTo(to_sr)
    for ii, geom in enumerate(geomcol):
        value_col[ii] = geom.GetX()
        value_row[ii] = geom.GetY()


def get_polygon_geometry_array(grid, fill):
    is_vectorized = grid.is_vectorized

    if grid.has_bounds:
        # We want geometries for everything even if masked.
        x_bounds = grid.x.bounds.value
        y_bounds = grid.y.bounds.value
        range_row = range(grid.shape[0])
        range_col = range(grid.shape[1])
        if is_vectorized:
            for row, col in itertools.product(range_row, range_col):
                min_x, max_x = np.min(x_bounds[col, :]), np.max(x_bounds[col, :])
                min_y, max_y = np.min(y_bounds[row, :]), np.max(y_bounds[row, :])
                polygon = box(min_x, min_y, max_x, max_y)
                fill[row, col] = polygon
        else:
            # tdk: we should be able to avoid the creation of this corners array
            corners = np.vstack((y_bounds, x_bounds))
            corners = corners.reshape([2] + list(x_bounds.shape))
            for row, col in itertools.product(range_row, range_col):
                current_corner = corners[:, row, col]
                coords = np.hstack((current_corner[1, :].reshape(-1, 1),
                                    current_corner[0, :].reshape(-1, 1)))
                polygon = Polygon(coords)
                fill[row, col] = polygon
    else:
        msg = 'A grid must have bounds/corners to construct polygons. Consider using "set_extrapolated_bounds".'
        raise GridDeficientError(msg)

    return fill


def get_point_geometry_array(grid, fill):
    """Create geometries for all the underlying coordinates regardless if the data is masked."""

    x_data = grid.x.value
    y_data = grid.y.value
    is_vectorized = grid.is_vectorized

    for idx_row, idx_col in itertools.product(*[range(ii) for ii in grid.shape]):
        if is_vectorized:
            y = y_data[idx_row]
            x = x_data[idx_col]
        else:
            y = y_data[idx_row, idx_col]
            x = x_data[idx_row, idx_col]
        pt = Point(x, y)
        fill[idx_row, idx_col] = pt
    return fill


def get_geometry_variable(grid, value=None, mask=None, use_bounds=True):
    is_empty = grid.is_empty
    if is_empty:
        mask = None
        value = None
    else:
        if mask is None:
            mask = grid.get_mask()
        if value is None:
            gp = GridGeometryProcessor(grid, None, mask, use_bounds=use_bounds)
            itr = gp.get_geometry_iterable()
            value = np.zeros(grid.shape, dtype=object)
            for idx, geometry in itr:
                if geometry is not None:
                    value[idx] = geometry
    if grid.abstraction == 'point':
        name = grid._point_name
    else:
        name = grid._polygon_name
    ret = GeometryVariable(name=name, value=value, mask=mask, attrs={'axis': 'geom'}, is_empty=is_empty,
                           dimensions=grid.dimensions, crs=grid.crs)
    return ret


def get_arr_intersects_bounds(arr, lower, upper, keep_touches=True):
    assert lower <= upper

    if keep_touches:
        arr_lower = arr >= lower
        arr_upper = arr <= upper
    else:
        arr_lower = arr > lower
        arr_upper = arr < upper

    ret = np.logical_and(arr_lower, arr_upper)
    return ret


def grid_update_mask(grid, bounds_sequence, keep_touches=True):
    minx, miny, maxx, maxy = bounds_sequence

    res_x = get_coordinate_boolean_array(grid.x, keep_touches, maxx, minx)
    res_y = get_coordinate_boolean_array(grid.y, keep_touches, maxy, miny)

    try:
        res = np.invert(np.logical_and(res_x.reshape(*grid.shape), res_y.reshape(*grid.shape)))
        if np.all(res):
            raise AllElementsMaskedError
        grid.set_mask(res)
    except AllElementsMaskedError:
        raise EmptySubsetError('grid')


def remove_nones(target):
    ret = filter(lambda x: x is not None, target)
    return ret


def get_extent_global(grid, comm=None):
    comm, rank, size = get_standard_comm_state(comm=comm)

    extent = grid.extent
    extents = comm.gather(extent)

    if rank == 0:
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
    ret = comm.bcast(ret)

    return ret


def get_coordinate_boolean_array(grid_target, keep_touches, max_target, min_target):
    target_centers = grid_target.value

    res_target = np.array(get_arr_intersects_bounds(target_centers, min_target, max_target, keep_touches=keep_touches))
    res_target = res_target.reshape(-1)

    return res_target


def get_hint_mask_from_geometry_bounds(grid, bbox, invert=True):
    grid_x = grid.x.value
    grid_y = grid.y.value

    minx, miny, maxx, maxy = bbox

    select_x = np.logical_and(grid_x >= minx, grid_x <= maxx)
    select_y = np.logical_and(grid_y >= miny, grid_y <= maxy)

    if grid.is_vectorized:
        select_y_expanded = np.zeros(grid.shape, dtype=bool)
        select_x_expanded = np.zeros(grid.shape, dtype=bool)
        for ii in range(grid.shape[0]):
            select_y_expanded[ii, :] = select_y[ii]
        for jj in range(grid.shape[1]):
            select_x_expanded[:, jj] = select_x[jj]
        select_x, select_y = select_x_expanded, select_y_expanded

    select = np.logical_and(select_x, select_y)

    if invert:
        select = np.invert(select)

    return select


def grid_set_geometry_variable_on_parent(func, grid, name, alloc_only=False):
    dimensions = [d.name for d in grid.dimensions]
    ret = get_geometry_variable(func, grid, name=name, attrs={'axis': 'geom'}, alloc_only=alloc_only,
                                dimensions=dimensions)
    return ret


def grid_set_mask_cascade(grid):
    members = grid.get_member_variables(include_bounds=True)
    grid.parent.set_mask(grid.mask_variable, exclude=members)


def expand_grid(grid):
    y = grid.parent[grid._y_name]
    x = grid.parent[grid._x_name]
    grid_is_empty = grid.is_empty

    if y.ndim == 1:
        if y.has_bounds:
            if not grid_is_empty:
                original_y_bounds = y.bounds.value
                original_x_bounds = x.bounds.value
            original_bounds_dimension_name = y.bounds.dimensions[-1].name
            has_bounds = True
            name_y = y.bounds.name
            name_x = x.bounds.name
        else:
            has_bounds = False

        if not grid_is_empty:
            new_x_value, new_y_value = np.meshgrid(x.value, y.value)
        new_dimensions = [y.dimensions[0], x.dimensions[0]]

        x.set_bounds(None)
        x.set_value(None)
        x.set_dimensions(new_dimensions)
        if not grid_is_empty:
            x.set_value(new_x_value)

        y.set_bounds(None)
        y.set_value(None)
        y.set_dimensions(new_dimensions)
        if not grid_is_empty:
            y.set_value(new_y_value)

        if has_bounds:
            grid._original_bounds_dimension_name = original_bounds_dimension_name

            new_y_bounds = np.zeros((original_y_bounds.shape[0], original_x_bounds.shape[0], 4),
                                    dtype=original_y_bounds.dtype)
            new_x_bounds = new_y_bounds.copy()
            for idx_y, idx_x in itertools.product(range(original_y_bounds.shape[0]), range(original_x_bounds.shape[0])):
                new_y_bounds[idx_y, idx_x, 0:2] = original_y_bounds[idx_y, 0]
                new_y_bounds[idx_y, idx_x, 2:4] = original_y_bounds[idx_y, 1]

                new_x_bounds[idx_y, idx_x, 0] = original_x_bounds[idx_x, 0]
                new_x_bounds[idx_y, idx_x, 1] = original_x_bounds[idx_x, 1]
                new_x_bounds[idx_y, idx_x, 2] = original_x_bounds[idx_x, 1]
                new_x_bounds[idx_y, idx_x, 3] = original_x_bounds[idx_x, 0]

            new_bounds_dimensions = new_dimensions + [Dimension('corners', size=4)]
            y.set_bounds(Variable(name=name_y, value=new_y_bounds, dimensions=new_bounds_dimensions, dist=y.dist,
                                  ranks=y.ranks))
            x.set_bounds(Variable(name=name_x, value=new_x_bounds, dimensions=new_bounds_dimensions, dist=x.dist,
                                  ranks=x.ranks))

    assert y.ndim == 2
    assert x.ndim == 2


def reorder_array(reorder_indices, arr, arr_dimensions, reorder_dimension, varying_dimension):
    """
    :param reorder_indices: Sequence of shift indices with same dimension as ``varying_dimension``. The shift index is a
     single integer value. Values in ``arr`` having indices >= to the shift index or translated such that the shift
     index in the original array is now zero in the reordered array.
    :type reorder_indices: sequence of integers
    :param arr: Array to reorder.
    :type arr: :class:`numpy.core.multiarray.ndarray`
    :param arr_dimensions: Dimension names for ``arr``.
    :type arr_dimensions: sequence of strings
    :param str reorder_dimension: The dimension in ``arr`` to reorder.
    :param str varying_dimension: The dimension in ``arr`` across which the ``reorder_dimension`` varies.
    :return: An in-place reordered array.
    :rtype: :class:`numpy.core.multiarray.ndarray`
    """

    reorder_index = arr_dimensions.index(reorder_dimension)
    varying_index = arr_dimensions.index(varying_dimension)

    itrs = [None] * arr.ndim
    for idx in range(len(arr.shape)):
        if idx == reorder_index:
            itrs[idx] = [slice(None)]
        else:
            itrs[idx] = range(arr.shape[idx])

    for yld in itertools.product(*itrs):
        curr_varying_index = yld[varying_index]
        the_split_index = reorder_indices[curr_varying_index]
        view_to_reorder = arr.__getitem__(yld)
        original_to_reorder = view_to_reorder.copy()
        offset = view_to_reorder.shape[0] - the_split_index
        view_to_reorder[0:offset] = original_to_reorder[the_split_index:]
        view_to_reorder[offset:] = original_to_reorder[0:the_split_index]
