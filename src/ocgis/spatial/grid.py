import abc
import itertools
from collections import OrderedDict

import numpy as np
from ocgis.driver.dimension_map import is_bounded, has_bounds

import ocgis
import six
from ocgis import Variable, vm
from ocgis.base import get_dimension_names, raise_if_empty, AbstractOcgisObject, get_variable_names, \
    is_unstructured_driver
from ocgis.constants import WrappedState, VariableName, KeywordArgument, GridAbstraction, DriverKey, \
    GridChunkerConstants, RegriddingRole, Topology, DMK, CFName
from ocgis.environment import ogr, env
from ocgis.exc import GridDeficientError, EmptySubsetError, AllElementsMaskedError
from ocgis.spatial.base import AbstractXYZSpatialContainer
from ocgis.spatial.geomc import AbstractGeometryCoordinates, PolygonGC, PointGC, LineGC
from ocgis.util.helpers import get_formatted_slice, get_iter, wrap_get_value, is_xarray, get_bounds_from_1d, \
    get_extrapolated_corners_esmf, create_ocgis_corners_from_esmf_corners
from ocgis.variable.base import get_dslice, get_dimension_lengths, is_empty
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable, get_masking_slice, GeometryProcessor
from ocgis.vmachine.mpi import MPI_SIZE
from shapely.geometry import Polygon, Point, box
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

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
            x_data = wrap_get_value(grid.x)
            y_data = wrap_get_value(grid.y)
            for idx_row, idx_col in itertools.product(*[list(range(ii)) for ii in grid.shape]):
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
        elif abstraction == GridAbstraction.POLYGON:
            if grid.has_bounds:
                # We want geometries for everything even if masked.
                xv = wrap_get_value(grid.parent[grid.x.bounds])
                yv = wrap_get_value(grid.parent[grid.y.bounds])

                range_row = list(range(grid.shape[0]))
                range_col = list(range(grid.shape[1]))
                if is_vectorized:
                    for row, col in itertools.product(range_row, range_col):
                        if hint_mask is not None and hint_mask[row, col]:
                            polygon = None
                        else:
                            min_x, max_x = np.min(xv[col, :]), np.max(xv[col, :])
                            min_y, max_y = np.min(yv[row, :]), np.max(yv[row, :])
                            polygon = box(min_x, min_y, max_x, max_y)
                        yield (row, col), polygon
                else:
                    # TODO: We should be able to avoid the creation of this corners array.
                    corners = np.vstack((yv, xv))
                    corners = corners.reshape([2] + list(xv.shape))
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


@six.add_metaclass(abc.ABCMeta)
class AbstractGrid(AbstractOcgisObject):
    """
    Base class for grid objects.

    :param abstraction: The grid abstraction to use. If ``'auto'`` (the default), use the highest order abstraction
     available.
    :type abstraction: :attr:`ocgis.constants.Topology`
    """

    def __init__(self, abstraction=KeywordArgument.Defaults.ABSTRACTION):
        if abstraction is None:
            raise ValueError("'abstraction' may not be None.")
        self.abstraction = abstraction

    @property
    def abstraction(self):
        """
        Get or set the spatial abstraction for the grid.

        :param abstraction: The grid's overloaded or highest order topology or spatial abstraction.
         :attr:`~ocgis.constants.Topology.AUTO` should not be returned.
        :type abstraction: :attr:`~ocgis.constants.Topology`.
        :rtype: :attr:`~ocgis.constants.Topology`
        """
        abstraction = self.dimension_map.get_grid_abstraction()
        if abstraction == Topology.AUTO:
            abstraction = self._get_auto_abstraction_()
        return abstraction

    @abstraction.setter
    def abstraction(self, abstraction):
        if not abstraction == Topology.AUTO:
            self.dimension_map.set_grid_abstraction(abstraction)

    @property
    def abstractions_available(self):
        """
        Get the topologies / spatial abstractions available on the object. Tuple elements are of type
        :attr:`ocgis.constants.Topology`.

        :rtype: tuple
        """
        return self._get_available_abstractions_()

    @abc.abstractmethod
    def get_abstraction_geometry(self, **kwargs):
        """
        Get the abstraction geometry variable for the grid.

        :param kwargs: Keyword arguments to the geometry get method. See :meth:`~ocgis.Grid.get_point` for example.
        :rtype: :class:`~ocgis.GeometryVariable`
        """
        raise NotImplementedError

    def is_abstraction_available(self, abstraction):
        """
        Return ``True`` if the spatial abstraction is available on the grid.

        :param abstraction: The spatial abstraction to check.
        :type abstraction: :attr:`ocgis.constants.GridAbstraction`
        :rtype: bool
        """
        return abstraction in self._get_available_abstractions_()

    @abc.abstractmethod
    def _get_auto_abstraction_(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_available_abstractions_(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _gc_create_index_bounds_(self, *args, **kwargs):
        raise NotImplementedError


class Grid(AbstractGrid, AbstractXYZSpatialContainer):
    """
    Grids are structured, rectilinear x/y-coordinate representations. x/y-coordinate variables may have bounds. The
    z-coordinate is supported only to allow its access from the grid. All subsetting operations, slicing, etc.
    occurs only on the x/y-coordinates.

    :param x: The grid's x-coordinate.
    :type x: :class:`~ocgis.Variable`
    :param y: The grid's y-coordinate.
    :type y: :class:`~ocgis.Variable`
    :param z: The grid's z-coordinate. No grid operations manipulate the z-coordinate. It is present on the grid for
     convenience.
    :type z: :class:`~ocgis.Variable`
    :param str abstraction: The grid's spatial abstraction.

    ============= ======================================================================================================
    Value         Description
    ============= ======================================================================================================
    ``'auto'``    Automatically choose spatial abstraction. `'polygon'` if x/y-coordinates have bounds and `'point'` if
                  they do not.
    ``'point'``   Use representative value from x/y-coordinate variables to construct point geometries. Typically this
                  is considered the center value.
    ``'polygon'`` Use bounds from x/y-coordinates to construct polygon geometries.
    ============= ======================================================================================================

    :param crs: See :class:`~ocgis.variable.geom.AbstractSpatialObject`
    :param parent: The parent field for the grid.
    :type parent: :class:`~ocgis.Field`
    :param mask: The mask variable for the grid. Coordinate variables should not be masked. The mask must be managed
     independently. The mask variable should use the its mask to indicate masked values.
    :param sequence pos: If coordinate variables ``x`` and ``y`` are two-dimensional, these are the dimension indices
     for them in the grid's dimensions. Defaults to ``(0, 1)`` or `(y/latitude, x/longitude)`.
    :type mask: :class:`~ocgis.Variable`
    """
    # TODO: Should this respond to level (i.e. ndim=3)?
    ndim = 2
    _point_name = VariableName.GEOMETRY_POINT
    _polygon_name = VariableName.GEOMETRY_POLYGON

    def __init__(self, x=None, y=None, z=None, pos=(0, 1), **kwargs):
        kwargs = kwargs.copy()
        kwargs[KeywordArgument.X] = x
        kwargs[KeywordArgument.Y] = y
        kwargs[KeywordArgument.Z] = z
        kwargs[KeywordArgument.POS] = pos
        abstraction = kwargs.pop(KeywordArgument.ABSTRACTION, KeywordArgument.Defaults.ABSTRACTION)

        # Structured grids are always considered isomorphic.
        kwargs[DMK.IS_ISOMORPHIC] = True

        AbstractXYZSpatialContainer.__init__(self, **kwargs)
        AbstractGrid.__init__(self, abstraction=abstraction)

        if env.SET_GRID_AXIS_ATTRS:
            self.x.attrs[CFName.AXIS] = 'X'
            self.y.attrs[CFName.AXIS] = 'Y'
            if self.z is not None:
                self.z.attrs[CFName.AXIS] = 'Z'

    def __getitem__(self, slc):
        """
        :param slc: The slice sequence with indices corresponding to:

        ===== ==================
        Index Description
        ===== ==================
        0     row/y dimension
        1     column/x dimension
        ===== ==================

        ``slc`` may also be a dictionary with grid dimensions as keys.

        :returns: Shallow copy of the sliced grid.
        :rtype: :class:`~ocgis.Grid`
        """
        if self.has_shared_dimension:
            raise ValueError('Grid coordinate variables may not have shared dimensions when slicing.')

        if not isinstance(slc, dict):
            slc = get_dslice(self.dimensions, slc)

        ret = self.copy()
        new_parent = ret.parent[slc]
        ret.parent = new_parent
        return ret

    def __setitem__(self, slc, grid):
        """
        Set the grid values and mask to match ``grid`` in the index space defined by ``slc``.

        :param slc: The set slice for the target. Must have length matching the grid dimension count.
        :type slc: `sequence` of :class:`slice`-like object
        :param grid: The grid object containing the values to set in the target.
        """

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

    @property
    def dtype(self):
        """
        :return: Representative data type for the grid. This is pulled from the archetype variable.
        :rtype: type
        """
        return self.archetype.dtype

    @property
    def has_allocated_abstraction_geometry(self):
        """
        :return: ``True`` if the geometry abstraction variable is allocated on the grid.
        :rtype: bool
        """
        if self.abstraction == 'point':
            return self.has_allocated_point
        elif self.abstraction == 'polygon':
            return self.has_allocated_polygon
        else:
            raise NotImplementedError(self.abstraction)

    @property
    def has_allocated_point(self):
        """
        :return: ``True`` if the point variable is allocated on the grid.
        :rtype: bool
        """
        if self._point_name in self.parent:
            return True
        else:
            return False

    @property
    def has_allocated_polygon(self):
        """
        :return: ``True`` if the polygon variable is allocated on the grid.
        :rtype: bool
        """
        if self._polygon_name in self.parent:
            return True
        else:
            return False

    @property
    def has_bounds(self):
        """
        Return ``True`` if the grid coordinate variables have bounds.

        :rtype: bool
        """
        return is_bounded(self.dimension_map, DMK.X)

    @property
    def has_shared_dimension(self):
        """Return ``True`` if the x/y dimensions are equal."""

        return len(set(get_dimension_names(self.dimensions))) != len(self.dimensions)

    @property
    def is_vectorized(self):
        """
        :return: ``True`` if the grid is vectorized (factorized). Vectorized grids have one-dimensional x- and
         coordinate variables.
        :rtype: bool
        """

        ndim = self.archetype.ndim
        if ndim == 1:
            ret = True
        else:
            ret = False
        return ret

    @property
    def shape(self):
        """
        :rtype: :class:`tuple` of :class:`int`
        """

        if is_xarray(self.archetype):
            if self.is_vectorized:
                ret = (self.y.shape[0], self.x.shape[0])
            else:
                ret = self.y.shape
        else:
            ret = get_dimension_lengths(self.dimensions)
        return ret

    def copy(self):
        """
        :return: shallow copy of the grid
        :rtype: :class:`~ocgis.Grid`
        """

        ret = super(AbstractGrid, self).copy()
        ret.parent = ret.parent.copy()
        return ret

    def diagnostics(self, plot_xy=False, scatter_xy=False, unique_xy=False, verbose=False, plot_var=None):
        """Print some grid diagnostics. This is designed to be customized."""

        assert self.is_vectorized

        xv = self.x.get_value()
        yv = self.y.get_value()

        x_unique = np.unique(xv)
        y_unique = np.unique(yv)

        lines = ['Structured Grid:']
        sub = ['        shape= {}'.format(self.shape),
               '         size= {}'.format(np.product(self.shape)),
               '    variables= {}'.format(get_variable_names([self.y, self.x])),
               '   dimensions= {}'.format(get_dimension_names(self.dimensions)),
               'is_vectorized= {}'.format(self.is_vectorized),
               ' shape_unique= {}'.format((y_unique.size, x_unique.size))
               ]
        if unique_xy:
            assert self.shape[0] == self.shape[1]
            uset = set()
            for ii in range(self.shape[0]):
                if ii % 100 == 0:
                    if verbose:
                        print('{} of {}'.format(ii, self.shape[0]))
                uset.update([(xv[ii], yv[ii])])
            sub.append('    unique_xy= {}'.format(len(set(uset))))

        sub = [' | {}'.format(s) for s in sub]
        lines.extend(sub)
        for l in lines:
            print(l)
        print('')
        print('Dimension Map:')
        self.dimension_map.pprint(as_dict=True)

        if plot_xy:
            import matplotlib.pyplot as plt
            assert self.is_vectorized
            f, axarr = plt.subplots(2)
            for ii, t in enumerate(['x', 'y']):
                axarr[ii].plot(np.arange(self.shape[ii]), getattr(self, t).get_value())
                axarr[ii].set_ylabel(t)
            axarr[0].set_title('Factorized Grid Coordinate Values')
            axarr[1].set_xlabel('Index'.format(t))
            plt.show()
        if scatter_xy:
            import matplotlib.pyplot as plt
            plt.scatter(self.x.get_value(), self.y.get_value())
            plt.title('X/Y Scatter Plot')
            plt.xlabel('x Coordinate Value')
            plt.ylabel('y Coordinate Value')
            plt.show()
        if plot_var is not None:
            import matplotlib.pyplot as plt
            assert self.is_vectorized
            var = getattr(self, plot_var)
            plt.scatter(np.arange(var.shape[0]), var.get_value())
            plt.xlabel('Index')
            plt.ylabel('Value')
            plt.title('Value Plot: {}'.format(var.name))
            plt.show()

    def extract(self, clean_break=False):
        """
        Extract the grid from its parent collection.

        See :meth:`~ocgis.Variable.extract` for documentation.
        """

        new_parent = self.parent.copy()
        original_parent = self.parent
        members = get_variable_names(self.get_member_variables())
        for v in self.parent.values():
            if v.name not in members:
                new_parent.remove_variable(v, remove_bounds=False)
        if clean_break:
            to_remove = []
            for v in original_parent.values():
                if v.name in members:
                    to_remove.append(v.name)
            for tr in to_remove:
                original_parent.remove_variable(tr, remove_bounds=False)
        self.parent = new_parent

        return self

    def expand(self):
        """
        If the grid is vectorized/factorized (spatial coordinate represented using one-dimensional arrays), convert
        spatial coordinates to two-dimensional arrays. If the grid is already two-dimensional, pass through.
        """
        expand_grid(self)

    def get_abstraction_geometry(self, **kwargs):
        """
        Get the abstraction geometry variable for the grid.

        :param kwargs: Keyword arguments to the geometry get method. See :meth:`~ocgis.Grid.get_point` for example.
        :rtype: :class:`~ocgis.GeometryVariable`
        """

        if self.abstraction == GridAbstraction.POINT:
            ret = self.get_point(**kwargs)
        elif self.abstraction == GridAbstraction.POLYGON:
            ret = self.get_polygon(**kwargs)
        else:
            raise NotImplementedError(self.abstraction)
        return ret

    def get_distributed_slice(self, slc, **kwargs):
        """
        Slice the grid in parallel and return a shallow copy. This is collective across the current
        :class:`~ocgis.OcgVM`.

        :param slc: See :meth:`~ocgis.Grid.__getitem__`.
        :param kwargs: Keyword arguments to :meth:`~ocgis.Variable.get_distributed_slice`.
        :rtype: :class:`~ocgis.AbstractGrid`
        """
        # TODO: This should be generalized for grid objects and use a standardized set of dimensions.
        raise_if_empty(self)

        if self.has_shared_dimension:
            raise ValueError('Grid coordinate variables may not have shared dimensions when slicing.')

        ret = self.copy()
        if is_xarray(self.archetype):
            ret.parent._storage = ret.parent._storage[slc]
        else:
            ret = self.copy()
            if isinstance(slc, dict):
                y_slc = slc[self.dimension_names[0]]
                x_slc = slc[self.dimension_names[1]]
                slc = [y_slc, x_slc]

            if self.is_vectorized:
                dummy_var = Variable(name='__ocgis_dummy_var__', dimensions=ret.dimensions)
                ret.parent.add_variable(dummy_var)
                ret.parent = dummy_var.get_distributed_slice(slc, **kwargs).parent
                ret.parent.pop(dummy_var.name)
            else:
                ret.parent = self.archetype.get_distributed_slice(slc, **kwargs).parent

        return ret

    def get_nearest(self, *args, **kwargs):
        return self.get_abstraction_geometry().get_nearest(*args, **kwargs).parent.grid

    def get_point(self, value=None, mask=None):
        return get_geometry_variable(self, value=value, mask=mask, use_bounds=False)

    def get_polygon(self, value=None, mask=None):
        return get_geometry_variable(self, value=value, mask=mask, use_bounds=True)

    def get_spatial_index(self, *args, **kwargs):
        return self.get_abstraction_geometry().get_spatial_index(*args, **kwargs)

    def get_report(self):
        """
        :return: sequence of strings containing explanatory grid information
        :rtype: :class:`list` of :class:`str`
        """

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

    def get_spatial_subset_operation(self, spatial_op, subset_geom, return_slice=False, use_bounds='auto',
                                     original_mask=None,
                                     keep_touches='auto', cascade=True, optimized_bbox_subset=False, apply_slice=True):
        """
        Perform intersects or intersection operations on the grid object.

        :param str spatial_op: Either an ``'intersects'`` or an ``'intersection'`` spatial operation.
        :param subset_geom: The subset Shapely geometry. All geometry types are accepted.
        :type subset_geom: :class:`shapely.geometry.base.BaseGeometry`
        :param bool return_slice: If ``True``, also return the slices used to limit the grid's extent.
        :param use_bounds: If ``'auto'`` (the default), use bounds if they are available to construct polygon objects
         for the intersects operation.
        :type use_bounds: :class:`bool` | :class:`str`
        :param original_mask: An optional mask to use as a hint for spatial operation. ``True`` values are excluded
         from spatial consideration.
        :type original_mask: :class:`numpy.ndarray`
        :param keep_touches: If ``'auto'`` (the default), keep geometries that touch only if the grid's spatial
         abstraction is point.
        :type keep_touches: :class:`bool` | :class:`str`
        :param cascade: If ``True`` (the default), set the mask across all variables in the grid's parent collection.
        :param optimized_bbox_subset: If ``True``, perform an optimized bounding box subset on the grid. This will only
         use the grid's representative coordinates ignoring bounds, geometries, etc.
        :param apply_slice: If ``True`` (the default), apply the slice to the grid object in addition to updating its
         mask.
        :return: If ``return_slice`` is ``False`` (the default), return a shallow copy of the sliced grid. If
         ``return_slice`` is ``True``, this will be a tuple with the subsetted object as the first element and the slice
         used as the second. If ``spatial_op`` is ``'intersection'``, the returned object is a geometry variable.
        :rtype: :class:`~ocgis.Grid` | :class:`~ocgis.GeometryVariable` | :class:`tuple` of ``(<returned object>, <slice used>)``
        """
        # TODO: This should be merged with geometry coordinates spatial subset operation.
        raise_if_empty(self)

        # try:
        subset_geom.prepare()
        # except AttributeError:
        #     if not isinstance(subset_geom, BaseGeometry):
        #         msg = 'Only Shapely geometries allowed for subsetting. Subset type is "{}".'.format(
        #             type(subset_geom))
        #         raise ValueError(msg)
        # else:
        subset_geom = subset_geom.get_value()[0]

        # Flag indicating presence of mask on grid prior to subsetting. If there is a mask, we always want to maintain
        # it. If not, only add a mask if some values will be masked.
        if self.get_mask() is None:
            original_grid_has_mask = False
        else:
            original_grid_has_mask = True

        if use_bounds is True and self.abstraction == 'point':
            msg = '"use_bounds" is True and grid abstraction is "point". Only a polygon abstraction may use bounds ' \
                  'during a spatial subset operation.'
            raise ValueError(msg)

        if not isinstance(subset_geom, BaseGeometry):
            msg = 'Only Shapely geometries allowed for subsetting. Subset type is "{}".'.format(
                type(subset_geom))
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
            if not optimized_bbox_subset:
                buffer_value = self.resolution * 1.25

            if isinstance(subset_geom, BaseMultipartGeometry):
                geom_itr = subset_geom
            else:
                geom_itr = [subset_geom]

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

        # tdk: HACK: it's not clear if xarray needs a copy here
        # ret = self.copy()
        ret = self

        if original_grid_has_mask:
            ret.set_mask(ret.get_mask().copy())

        if optimized_bbox_subset:
            if original_mask is not None and original_mask.any():
                # TODO: OPTIMIZE: Can we avoid the cascade? There is no reason to cascade the mask if it is going to be sliced off anyway.
                # tdk: FIX: how to deal with masking and xarray?
                # ret.set_mask(original_mask, cascade=True)
                pass
            sliced_grid, _, the_slice = get_masking_slice(original_mask, ret, apply_slice=apply_slice)
            # tdk: FIX: how to deal with masking and xarray
            # sliced_grid_mask = sliced_grid.get_mask()
            # if not original_grid_has_mask and sliced_grid_mask is not None and not sliced_grid_mask.any():
            #     sliced_grid.set_mask(None)
        else:
            fill_mask = original_mask
            geometry_fill = None
            # If everything is masked, there is no reason to load the grid geometries.
            if not original_mask.all():
                if perform_intersection:
                    geometry_fill = np.zeros(fill_mask.shape, dtype=object)
                if vm.size > 1:
                    new_intersects_target = subset_geom.intersection(box(*self.extent).buffer(1e-6))
                else:
                    new_intersects_target = subset_geom
                gp = GridGeometryProcessor(self, new_intersects_target, original_mask, keep_touches=keep_touches,
                                           use_bounds=use_bounds)
                for idx, intersects_logical, current_geometry in gp.iter_intersects():
                    fill_mask[idx] = not intersects_logical
                    if perform_intersection and intersects_logical:
                        geometry_fill[idx] = current_geometry.intersection(subset_geom)

            if perform_intersection:
                if geometry_fill is None:
                    if use_bounds:
                        name = self._polygon_name
                    else:
                        name = self._point_name
                    geometry_variable = GeometryVariable(name=name)
                else:
                    if use_bounds:
                        geometry_variable = ret.get_polygon(value=geometry_fill, mask=fill_mask)
                    else:
                        geometry_variable = ret.get_point(value=geometry_fill, mask=fill_mask)
                ret.parent.add_variable(geometry_variable, force=True)

            sliced_grid, sliced_mask, the_slice = get_masking_slice(fill_mask, ret, apply_slice=apply_slice)

            # Only modify the outgoing mask if any values are masked.
            sliced_mask_value = sliced_mask.get_value()
            if sliced_mask_value is not None and sliced_mask_value.any():
                sliced_grid.set_mask(sliced_mask_value, cascade=cascade)

        if perform_intersection:
            obj_to_ret = sliced_grid.parent[geometry_variable.name]
        else:
            obj_to_ret = sliced_grid

        if return_slice:
            ret = (obj_to_ret, the_slice)
        else:
            ret = obj_to_ret

        return ret

    def get_value_stacked(self):
        y = wrap_get_value(self.y)
        x = wrap_get_value(self.x)

        if self.is_vectorized:
            x, y = np.meshgrid(x, y)

        fill = np.zeros([2] + list(y.shape))
        fill[0, :, :] = y
        fill[1, :, :] = x
        return fill

    def iter_records(self, *args, **kwargs):
        return self.get_abstraction_geometry().iter_records(*args, **kwargs)

    def remove_bounds(self):
        """Set the grid coordinate variable bounds to ``None``."""

        self.x.set_bounds(None)
        self.y.set_bounds(None)
        dmap = self.dimension_map
        dmap.set_bounds(DMK.X, None)
        dmap.set_bounds(DMK.Y, None)

    def reorder(self):
        if self.wrapped_state != WrappedState.WRAPPED:
            raise ValueError('Only wrapped coordinates may be reordered.')

        if self.is_empty:
            return self

        reorder_dimension = self.dimensions[1].name
        varying_dimension = self.dimensions[0].name

        if self.dimensions[1].dist and vm.size > 1:
            raise ValueError('The reorder dimension may not be distributed.')

        # Reorder indices identify where the index translation occurs.
        wrapped = self.x.get_value()

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
            reorder_array(shift_indices, self.x.get_value().reshape(1, -1), get_dimension_names(self.dimensions),
                          reorder_dimension, varying_dimension)

        # Reorder all arrays that have the reorder and varying dimension.
        for var in list(self.parent.values()):
            arr_dimension_names = get_dimension_names(var.dimensions)
            if reorder_dimension in arr_dimension_names and varying_dimension in arr_dimension_names:
                reorder_array(shift_indices, var.get_value(), arr_dimension_names, reorder_dimension, varying_dimension)
                if var.has_masked_values:
                    mask = var.get_mask()
                    if mask is not None:
                        reorder_array(shift_indices, mask, arr_dimension_names, reorder_dimension, varying_dimension)
                        var.set_mask(mask)

    def set_extrapolated_bounds(self, name_x_variable, name_y_variable, name_dimension):
        """
        Extrapolate corners from grid centroids.

        :param str name_x_variable: Name for the x-coordinate bounds variable.
        :param str name_y_variable: Name for the y-coordinate bounds variable.
        :param str name_dimension: Name for the bounds/corner dimension.
        """

        if is_xarray(self.archetype):
            import xarray as xr
            names = [name_y_variable, name_x_variable]

            for ii, target in enumerate([self.y, self.x]):
                values = target.values
                if target.ndim == 1:
                    bounds_value = get_bounds_from_1d(values)
                    new_dims = list(target.dims)
                else:
                    # TODO: consider renaming this functions to get_bounds_from_2d.
                    bounds_value = get_extrapolated_corners_esmf(values)
                    bounds_value = create_ocgis_corners_from_esmf_corners(bounds_value)
                    new_dims = list(self.dims)
                new_dims.append(name_dimension)
                to_add = xr.DataArray(bounds_value, name=names[ii], dims=new_dims)
                self.parent.add_variable(to_add)
        else:
            self.x.set_extrapolated_bounds(name_x_variable, name_dimension)
            self.y.set_extrapolated_bounds(name_y_variable, name_dimension)
            self.parent = self.y.parent

    def update_crs(self, *args, **kwargs):
        self.expand()
        super(AbstractGrid, self).update_crs(*args, **kwargs)

    def write(self, *args, **kwargs):
        """
        See :meth:`~ocgis.Field.write`.
        """
        self.parent.write(*args, **kwargs)

    def _get_auto_abstraction_(self):
        if self.has_bounds:
            ret = GridAbstraction.POLYGON
        else:
            ret = GridAbstraction.POINT
        return ret

    def _get_available_abstractions_(self):
        ret = [GridAbstraction.POINT]
        if has_bounds(self.archetype):
            ret.append(GridAbstraction.POLYGON)
        return ret

    def _get_canonical_dimension_map_(self, field=None, create=False):
        if field is None:
            field = self.parent
        return field.dimension_map

    def _get_dimensions_(self):
        """
        Grid dimensions are always:

        ===== ==================
        Index Description
        ===== ==================
        0     row/y-dimension
        1     column/x-dimension
        ===== ==================

        See superclass for additional documentation.
        """
        dmap = self._get_canonical_dimension_map_()
        ydim_name = dmap.get_dimension(DMK.Y)[0]
        xdim_name = dmap.get_dimension(DMK.X)[0]
        if is_xarray(self.archetype):
            ret = (ydim_name, xdim_name)
        else:
            dimensions = self.parent.dimensions
            ret = (dimensions[ydim_name], dimensions[xdim_name])
        return ret

    def _get_extent_(self):
        if is_empty(self):
            return None

        has_bounds = is_bounded(self.parent.dimension_map, DMK.X)
        if has_bounds:
            x = wrap_get_value(self.parent[self.x.bounds])
            y = wrap_get_value(self.parent[self.y.bounds])
        else:
            x = wrap_get_value(self.x)
            y = wrap_get_value(self.y)

        minx = x.min()
        miny = y.min()
        maxx = x.max()
        maxy = y.max()

        return minx, miny, maxx, maxy

    def _get_is_empty_(self):
        return is_empty(self.parent)

    @staticmethod
    def _gc_create_global_indices_(global_shape, **kwargs):
        return np.arange(1, six.moves.reduce(lambda x, y: x * y, global_shape) + 1, **kwargs).reshape(*global_shape,
                                                                                                      order='C')

    def _gc_create_index_bounds_(self, regridding_role, host_attribute_variable, parent, slices, split_dimension,
                                 bounds_dimension):
        raise_if_empty(self)

        constants_gci = GridChunkerConstants.IndexFile
        if regridding_role == RegriddingRole.DESTINATION:
            name_x_bounds = constants_gci.NAME_X_DST_BOUNDS_VARIABLE
            name_y_bounds = constants_gci.NAME_Y_DST_BOUNDS_VARIABLE
            erole_x = constants_gci.ESMF_ROLE_DST_BOUNDS_X
            erole_y = constants_gci.ESMF_ROLE_DST_BOUNDS_Y
        elif regridding_role == RegriddingRole.SOURCE:
            name_x_bounds = constants_gci.NAME_X_SRC_BOUNDS_VARIABLE
            name_y_bounds = constants_gci.NAME_Y_SRC_BOUNDS_VARIABLE
            erole_x = constants_gci.ESMF_ROLE_SRC_BOUNDS_X
            erole_y = constants_gci.ESMF_ROLE_SRC_BOUNDS_Y
        else:
            raise NotImplementedError(regridding_role)

        host_attribute_variable.attrs[name_x_bounds] = name_x_bounds
        host_attribute_variable.attrs[name_y_bounds] = name_y_bounds

        xb = Variable(name=name_x_bounds, dimensions=[split_dimension, bounds_dimension],
                      attrs={'esmf_role': erole_x},
                      dtype=env.NP_INT)
        yb = Variable(name=name_y_bounds, dimensions=[split_dimension, bounds_dimension],
                      attrs={'esmf_role': erole_y},
                      dtype=env.NP_INT)
        x_name = self.x.dimensions[0].name
        y_name = self.y.dimensions[0].name
        for idx, slc in enumerate(slices):
            xb.get_value()[idx, :] = slc[x_name].start, slc[x_name].stop
            yb.get_value()[idx, :] = slc[y_name].start, slc[y_name].stop
        parent.add_variable(xb)
        parent.add_variable(yb)

    def _gc_initialize_(self, regridding_role):
        pass

    def _gc_nchunks_dst_(self, grid_chunker):
        try:
            ret = super(Grid, self)._gc_nchunks_dst_(grid_chunker)
        except NotImplementedError:
            if self.ndim != 2:
                raise NotImplementedError('Only implemented for two dimensions.')
            else:
                ret = (10, 10)
        return ret

    def _initialize_parent_(self, *args, **kwargs):
        return self._get_parent_class_()(*args, **kwargs)


class GridUnstruct(AbstractGrid):
    """
    Unstructured grids manage operations across geometry coordinate objects. It overloads some operations but generally
    delegates complex operations to underlying geometry coordinate objects. It will broadcast operations across multiple
    geometry coordinate objects as necessary. Hence, geometry coordinate objects' documentation should be used when
    interpreting unstructured grid operations.

    :param geoms: One or more geometry coordinate variables for representing the unstructured grid. If ``None``, use the
     parent's dimension map to construct the object.
    :type geoms: sequence of :class:`~ocgis.spatial.geomc.AbstractGeometryCoordinates` | None
    :param abstraction: See :attr:`ocgis.spatial.grid.AbstractGrid.abstraction`.
    :param parent: The parent field object. Required if not geometry coordinate objects are provided.
    :type parent: :class:`~ocgis.Field`
    """

    __internal_attrs__ = ('__customizers__', 'abstraction', 'abstractions_available', 'archetype',
                          'coordinate_variables', 'dimension_map', 'geoms', 'get_abstraction_geometry', 'reduce_global',
                          'update_crs', '_get_auto_abstraction_', '_get_available_abstractions_',
                          '_gc_create_index_bounds_')
    __customizers__ = {Topology.POLYGON: PolygonGC, Topology.LINE: LineGC, Topology.POINT: PointGC}

    def __init__(self, geoms=None, abstraction=GridAbstraction.AUTO, parent=None):
        if geoms is None:
            dimension_map = parent.dimension_map
            if dimension_map.has_topology:
                topologies = dimension_map.get_available_topologies()
                c = self.__customizers__
                poss = [c[p] for p in c.keys() if p in topologies]
            else:
                poss = []
            geoms = [p(parent=parent) for p in poss]

            if len(geoms) == 0:
                raise GridDeficientError('Cannot construct any geometry coordinate objects from parent')

        self.geoms = tuple(get_iter(geoms, dtype=AbstractGeometryCoordinates))

        # Quick test to make sure the parents of the geometry coordinate objects are the same reference.
        geoms = self.geoms
        t = id(geoms[0].parent)
        for g in geoms:
            assert t == id(g.parent)

        # Let the geometry coordinate objects know they are hosted. They will return grids instead of themselves
        # following the return format decorator.
        for g in geoms:
            g.hosted = True

        # Always overload the driver to UGRID if the current driver is not unstructured.
        driver_klass = self.dimension_map.get_driver(as_class=True)
        if not is_unstructured_driver(driver_klass):
            self.dimension_map.set_driver(DriverKey.NETCDF_UGRID)

        super(GridUnstruct, self).__init__(abstraction=abstraction)

    def __getattribute__(self, name):
        if name == '__internal_attrs__' or name in self.__internal_attrs__:
            ret = object.__getattribute__(self, name)
        else:
            ret = getattr(self.archetype, name)
        return ret

    @property
    def archetype(self):
        hierarchy = [GridAbstraction.POLYGON, GridAbstraction.LINE, GridAbstraction.POINT]
        possible = {g.abstraction: g for g in self.geoms}
        if self.abstraction == GridAbstraction.AUTO:
            for h in hierarchy:
                try:
                    return possible[h]
                except KeyError:
                    continue
        else:
            return possible[self.abstraction]

    @property
    def coordinate_variables(self):
        """
        See :meth:`~ocgis.collection.field.Field.coordinate_variables`
        """
        ret = []
        for g in self.geoms:
            for c in g.coordinate_variables:
                if c.name not in ret:
                    ret.append(c)
        return tuple(ret)

    @property
    def dimension_map(self):
        return self.geoms[0].dimension_map

    def get_abstraction_geometry(self):
        arch = self.archetype

        geoms = [g[1] for g in arch.iter_geometries()]
        mask = self.get_mask()

        kwargs = {}
        kwargs[KeywordArgument.VALUE] = geoms
        kwargs[KeywordArgument.MASK] = mask
        kwargs[KeywordArgument.CRS] = self.crs
        kwargs[KeywordArgument.ATTRS] = {'axis': 'ocgis_geom'}
        kwargs[KeywordArgument.DIMENSIONS] = self.element_dim
        ret = GeometryVariable(**kwargs)
        return ret

    def reduce_global(self, *args, **kwargs):
        """See :meth:`ocgis.spatial.geomc.AbstractGeometryCoordinates.reduce_global`"""

        # Reductions are only important for higher-level abstractions (i.e. polygons). Polygons and lines should share
        # coordinate arrays.
        for a in self.abstractions_available.values():
            a = a.reduce_global(*args, **kwargs)
            return a.parent.grid

    def update_crs(self, *args, **kwargs):
        for g in self.geoms:
            g.update_crs(*args, **kwargs)

    def _get_auto_abstraction_(self):
        return list(self.abstractions_available.keys())[0]

    def _get_available_abstractions_(self):
        hierarchy = [GridAbstraction.POLYGON, GridAbstraction.LINE, GridAbstraction.POINT]
        possible = {g.abstraction: g for g in self.geoms}
        ret = OrderedDict()
        for h in hierarchy:
            if h in possible:
                ret[h] = possible[h]
        return ret

    def _gc_create_index_bounds_(self, *args, **kwargs):
        pass


def arr_intersects_bounds(arr, lower, upper, keep_touches=True, section_slice=None):
    assert lower <= upper

    if section_slice is not None:
        ret = np.zeros(arr.shape, dtype=bool)
        arr = arr[section_slice]

    if keep_touches:
        arr_lower = arr >= lower
        arr_upper = arr <= upper
    else:
        arr_lower = arr > lower
        arr_upper = arr < upper

    if section_slice is None:
        ret = np.logical_and(arr_lower, arr_upper)
    else:
        ret[section_slice] = np.logical_and(arr_lower, arr_upper)

    return ret


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
        x_bounds = grid.x.bounds.get_value()
        y_bounds = grid.y.bounds.get_value()
        range_row = list(range(grid.shape[0]))
        range_col = list(range(grid.shape[1]))
        if is_vectorized:
            for row, col in itertools.product(range_row, range_col):
                min_x, max_x = np.min(x_bounds[col, :]), np.max(x_bounds[col, :])
                min_y, max_y = np.min(y_bounds[row, :]), np.max(y_bounds[row, :])
                polygon = box(min_x, min_y, max_x, max_y)
                fill[row, col] = polygon
        else:
            # TODO: We should be able to avoid the creation of this corners array.
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

    x_data = grid.x.get_value()
    y_data = grid.y.get_value()
    is_vectorized = grid.is_vectorized

    for idx_row, idx_col in itertools.product(*[list(range(ii)) for ii in grid.shape]):
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
    ret = GeometryVariable(name=name, value=value, mask=mask, attrs={'axis': 'ocgis_geom'}, dimensions=grid.dimensions,
                           crs=grid.crs)

    # tdk: FEATURE: this needs to be done with the xarray driver i think
    ret = ret.to_xarray()

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
    ret = [x for x in target if x is not None]
    return ret


def get_coordinate_boolean_array(grid_target, keep_touches, max_target, min_target):
    target_centers = grid_target.get_value()

    res_target = np.array(arr_intersects_bounds(target_centers, min_target, max_target, keep_touches=keep_touches))
    res_target = res_target.reshape(-1)

    return res_target


def get_hint_mask_from_geometry_bounds(grid, bbox, invert=True):
    grid_x = wrap_get_value(grid.x)
    grid_y = wrap_get_value(grid.y)

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
    y = grid.y
    x = grid.x
    grid_is_empty = grid.is_empty

    if y.ndim == 1:
        if y.has_bounds:
            if not grid_is_empty:
                original_y_bounds = y.bounds.get_value()
                original_x_bounds = x.bounds.get_value()
            original_bounds_dimension_name = y.bounds.dimensions[-1].name
            has_bounds = True
            name_y = y.bounds.name
            name_x = x.bounds.name
        else:
            has_bounds = False

        if not grid_is_empty:
            new_x_value, new_y_value = np.meshgrid(x.get_value(), y.get_value())
        new_dimensions = [y.dimensions[0], x.dimensions[0]]

        x.set_bounds(None)
        x._value = None
        x.set_dimensions(new_dimensions)
        if not grid_is_empty:
            x.set_value(new_x_value)

        y.set_bounds(None)
        y._value = None
        y.set_dimensions(new_dimensions)
        if not grid_is_empty:
            y.set_value(new_y_value)

        if has_bounds:
            grid._original_bounds_dimension_name = original_bounds_dimension_name

            new_y_bounds = np.zeros((original_y_bounds.shape[0], original_x_bounds.shape[0], 4),
                                    dtype=original_y_bounds.dtype)
            new_x_bounds = new_y_bounds.copy()
            for idx_y, idx_x in itertools.product(list(range(original_y_bounds.shape[0])),
                                                  list(range(original_x_bounds.shape[0]))):
                new_y_bounds[idx_y, idx_x, 0:2] = original_y_bounds[idx_y, 0]
                new_y_bounds[idx_y, idx_x, 2:4] = original_y_bounds[idx_y, 1]

                new_x_bounds[idx_y, idx_x, 0] = original_x_bounds[idx_x, 0]
                new_x_bounds[idx_y, idx_x, 1] = original_x_bounds[idx_x, 1]
                new_x_bounds[idx_y, idx_x, 2] = original_x_bounds[idx_x, 1]
                new_x_bounds[idx_y, idx_x, 3] = original_x_bounds[idx_x, 0]

            new_bounds_dimensions = new_dimensions + [Dimension('corners', size=4)]
            y.set_bounds(Variable(name=name_y, value=new_y_bounds, dimensions=new_bounds_dimensions))
            x.set_bounds(Variable(name=name_x, value=new_x_bounds, dimensions=new_bounds_dimensions))

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
            itrs[idx] = list(range(arr.shape[idx]))

    for yld in itertools.product(*itrs):
        curr_varying_index = yld[varying_index]
        the_split_index = reorder_indices[curr_varying_index]
        view_to_reorder = arr.__getitem__(yld)
        original_to_reorder = view_to_reorder.copy()
        offset = view_to_reorder.shape[0] - the_split_index
        view_to_reorder[0:offset] = original_to_reorder[the_split_index:]
        view_to_reorder[offset:] = original_to_reorder[0:the_split_index]
