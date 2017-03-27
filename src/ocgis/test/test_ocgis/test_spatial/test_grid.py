import itertools
from copy import deepcopy

import fiona
import numpy as np
from numpy.testing.utils import assert_equal
from shapely import wkt
from shapely.geometry import Point, box, MultiPolygon, shape
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from ocgis import RequestDataset
from ocgis.base import get_variable_names
from ocgis.collection.field import OcgField
from ocgis.constants import KeywordArguments
from ocgis.driver.nc import DriverNetcdfCF
from ocgis.exc import EmptySubsetError, BoundsAlreadyAvailableError
from ocgis.spatial.grid import GridXY, expand_grid, GridGeometryProcessor
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.util.helpers import make_poly, iter_array
from ocgis.variable.base import Variable, VariableCollection, SourcedVariable
from ocgis.variable.crs import WGS84, CoordinateReferenceSystem, Spherical
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable
from ocgis.vm.mpi import MPI_RANK, MPI_COMM, variable_gather, MPI_SIZE, OcgMpi, variable_scatter


class Test(AbstractTestInterface):
    def test_expand_grid(self):
        x = [101, 102, 103]
        y = [40, 41, 42, 43]
        vx = Variable('x', value=x, dtype=float, dimensions='xdim')
        vx.set_extrapolated_bounds('x_bnds', 'bounds')
        vy = Variable('y', value=y, dtype=float, dimensions='ydim')
        vy.set_extrapolated_bounds('y_bnds', 'bounds')

        grid = GridXY(vx, vy)

        for variable in [vx, vy]:
            self.assertEqual(grid.parent[variable.name].ndim, 1)
        expand_grid(grid)

        for variable in [vx, vy]:
            self.assertEqual(grid.parent[variable.name].ndim, 2)


class TestGridGeometryProcessor(AbstractTestInterface):
    def test(self):

        keywords = {'with_xy_bounds': [False, True], 'add_hint_mask': [False, True], 'with_2d_variables': [False, True]}

        for k in self.iter_product_keywords(keywords, as_namedtuple=False):
            add_hint_mask = k.pop('add_hint_mask')

            grid = self.get_gridxy(**k)
            subset_geometry = box(*grid.extent).buffer(1.0)
            if add_hint_mask:
                hint_mask = np.ones(grid.shape, dtype=bool)
            else:
                hint_mask = None
            gp = GridGeometryProcessor(grid, subset_geometry, hint_mask)

            actual = list(gp.iter_intersects())
            self.assertEqual(len(actual), grid.shape[0] * grid.shape[1])
            for a in actual:
                if not add_hint_mask:
                    if k['with_xy_bounds']:
                        self.assertIsInstance(a[2], Polygon)
                    else:
                        self.assertIsInstance(a[2], Point)
                else:
                    self.assertIsNone(a[2])


class TestGridXY(AbstractTestInterface):
    def assertGridCorners(self, grid):
        """
        :type grid: :class:`ocgis.new_interface.grid.GridXY`
        """

        assert grid.corners is not None

        def _get_is_ascending_(arr):
            """
            Return ``True`` if the array is ascending from index 0 to -1.

            :type arr: :class:`numpy.ndarray`
            :rtype: bool
            """

            assert (arr.ndim == 1)
            if arr[0] < arr[-1]:
                ret = True
            else:
                ret = False

            return ret

        # Assert polygon constructed from grid corners contains the associated centroid value.
        for ii, jj in itertools.product(range(grid.shape[0]), range(grid.shape[1])):
            pt = Point(grid.value.data[1, ii, jj], grid.value.data[0, ii, jj])
            poly_corners = grid.corners.data[:, ii, jj]
            rtup = (poly_corners[0, :].min(), poly_corners[0, :].max())
            ctup = (poly_corners[1, :].min(), poly_corners[1, :].max())
            poly = make_poly(rtup, ctup)
            self.assertTrue(poly.contains(pt))

        # Assert masks are equivalent between value and corners.
        for (ii, jj), m in iter_array(grid.value.mask[0, :, :], return_value=True):
            if m:
                self.assertTrue(grid.corners.mask[:, ii, jj].all())
            else:
                self.assertFalse(grid.corners.mask[:, ii, jj].any())

        grid_y = grid._y
        grid_x = grid._x
        if grid_y is not None or grid_x is not None:
            self.assertEqual(_get_is_ascending_(grid_y.value), _get_is_ascending_(grid.corners.data[0, :, 0][:, 0]))
            self.assertEqual(_get_is_ascending_(grid_x.value), _get_is_ascending_(grid.corners.data[1, 0, :][:, 0]))

    def get_iter_gridxy(self, return_kwargs=False):
        poss = [True, False]
        kwds = dict(with_2d_variables=poss)
        for k in self.iter_product_keywords(kwds, as_namedtuple=False):
            ret = self.get_gridxy(**k)
            if return_kwargs:
                ret = (ret, k)
            yield ret

    def test_init(self):
        crs = WGS84()
        grid = self.get_gridxy(crs=crs)
        self.assertIsInstance(grid, GridXY)
        self.assertIn('x', grid.parent)
        self.assertIn('y', grid.parent)
        self.assertEqual(grid.crs, crs)
        self.assertEqual([dim.name for dim in grid.dimensions], ['ydim', 'xdim'])
        self.assertEqual(grid.shape, (4, 3))
        self.assertTrue(grid.is_vectorized)
        self.assertEqual(grid.x.ndim, 1)
        self.assertEqual(grid.y.ndim, 1)

        # Test with different variable names.
        x = Variable(name='col', value=[1], dimensions='col')
        y = Variable(name='row', value=[2], dimensions='row')
        grid = GridXY(x, y)
        assert_equal(grid.x.value, [1])
        assert_equal(grid.y.value, [2])

        # Test point and polygon representations.
        grid = self.get_gridxy(crs=WGS84())
        grid.set_extrapolated_bounds('x_bounds', 'y_bounds', 'bounds')
        targets = ['get_point', 'get_polygon']
        targets = [getattr(grid, t)() for t in targets]
        for t in targets:
            self.assertIsInstance(t, GeometryVariable)
        self.assertTrue(grid.is_vectorized)
        sub = grid[1, 1]
        targets = ['get_point', 'get_polygon']
        targets = [getattr(sub, t)() for t in targets]
        for t in targets:
            self.assertEqual(t.shape, (1, 1))
            self.assertIsInstance(t, GeometryVariable)
        self.assertTrue(grid.is_vectorized)

    def test_init_from_file(self):
        """Test loading from file."""

        grid = self.get_gridxy()
        path = self.get_temporary_file_path('foo.nc')
        grid.write(path)
        rd = RequestDataset(uri=path)
        x = SourcedVariable(name=grid.x.name, request_dataset=rd, protected=True)
        y = SourcedVariable(name=grid.y.name, request_dataset=rd, protected=True)
        self.assertIsNone(x._value)
        self.assertIsNone(y._value)
        fgrid = GridXY(x, y)
        self.assertEqual(len(fgrid.dimensions), 2)
        for target in [fgrid._y_name, fgrid._x_name]:
            fgrid.parent[target].protected = False
        actual = np.mean([fgrid.x.value.mean(), fgrid.y.value.mean()])
        self.assertEqual(actual, 71.75)

    def test_system_masking(self):
        """Test behavior of the grid mask. This is an independently managed variable."""

        x = Variable('xc', value=[1, 2, 3], dimensions='dimx')
        y = Variable('yc', value=[10, 20, 30, 40], dimensions='dimy')
        grid = GridXY(x, y)
        data = Variable('data', value=np.zeros(grid.shape), dimensions=['dimy', 'dimx'])
        grid.parent.add_variable(data)

        gmask = grid.get_mask()
        self.assertIsNone(gmask)
        self.assertIsNone(grid.mask_variable)

        new_mask = np.zeros(grid.shape, dtype=bool)
        new_mask[1, 1] = True
        grid.set_mask(new_mask, cascade=True)
        self.assertIsInstance(grid.mask_variable, Variable)
        actual = grid.get_mask()
        self.assertNumpyAll(actual, new_mask)
        actual = get_variable_names(grid.get_member_variables())
        desired = [x.name, y.name, grid._mask_name]
        self.assertAsSetEqual(actual, desired)
        self.assertNumpyAll(grid.get_mask(), data.get_mask())

        path = self.get_temporary_file_path('foo.nc')
        grid.parent.write(path)

        with self.nc_scope(path) as ds:
            actual = ds.variables[grid.mask_variable.name]
            self.assertNumpyAll(grid.get_mask(), actual[:].mask)
            # RequestDataset(uri=path).inspect()

        # Test mask is used when read from file.
        actual_field = RequestDataset(path).get()
        self.assertNumpyAll(grid.get_mask(), actual_field.grid.get_mask())
        self.assertEqual(actual_field.grid.get_mask().sum(), 1)
        self.assertTrue(actual_field.grid.is_vectorized)
        self.assertEqual(actual_field.grid.get_mask().dtype, bool)
        actual_field.set_abstraction_geom()
        self.assertNumpyAll(actual_field.geom.get_mask(), grid.get_mask())

    def test_copy(self):
        grid = self.get_gridxy()

        grid_copy = grid.copy()
        nuisance = Variable(name='nuisance')
        grid_copy.parent.add_variable(nuisance)
        self.assertNotIn(nuisance.name, grid.parent)

    @attr('mpi')
    def test_extent_global(self):
        desired = (-180.0, -90.0, 180.0, 90.0)
        grid = self.get_gridxy_global()
        actual = grid.extent_global
        self.assertEqual(actual, desired)

    def test_getitem(self):
        grid = self.get_gridxy()
        self.assertEqual(grid.ndim, 2)
        sub = grid[2, 1]
        self.assertNotIn('point', sub.parent)
        self.assertNotIn('polygon', sub.parent)
        self.assertEqual(sub.x.value, 102.)
        self.assertEqual(sub.y.value, 42.)

        # Test with two-dimensional x and y values.
        grid = self.get_gridxy(with_2d_variables=True)
        sub = grid[1:3, 1:3]
        actual_x = [[102.0, 103.0], [102.0, 103.0]]
        self.assertEqual(sub.x.value.tolist(), actual_x)
        actual_y = [[41.0, 41.0], [42.0, 42.0]]
        self.assertEqual(sub.y.value.tolist(), actual_y)

        # Test with parent.
        grid = self.get_gridxy(with_parent=True)
        self.assertEqual(id(grid.x.parent), id(grid.y.parent))
        orig_tas = grid.parent['tas'].value[slice(None), slice(1, 2), slice(2, 4)]
        orig_rhs = grid.parent['rhs'].value[slice(2, 4), slice(1, 2), slice(None)]
        self.assertEqual(grid.shape, (4, 3))

        sub = grid[2:4, 1]
        self.assertEqual(grid.shape, (4, 3))
        self.assertEqual(sub.parent['tas'].shape, (10, 1, 2))
        self.assertEqual(sub.parent['rhs'].shape, (2, 1, 10))
        self.assertNumpyAll(sub.parent['tas'].value, orig_tas)
        self.assertNumpyAll(sub.parent['rhs'].value, orig_rhs)
        self.assertTrue(np.may_share_memory(sub.parent['tas'].value, grid.parent['tas'].value))

    @attr('mpi', 'mpi-3')
    def test_get_distributed_slice(self):
        path = '/home/benkoziol/l/data/bekozi-work/esmf-optimizations/cr-grids/grid_cr_src.nc'
        rd = RequestDataset(path)
        grid = rd.get().grid
        bounds_global = deepcopy([d.bounds_global for d in grid.dimensions])

        for _ in range(10):
            sub = grid.get_distributed_slice([slice(73, 157), slice(305, 386)])
            bounds_global_grid_after_slice = [d.bounds_global for d in grid.dimensions]
            self.assertEqual(bounds_global, bounds_global_grid_after_slice)

    @attr('mpi')
    def test_get_gridxy(self):
        ret = self.get_gridxy()
        self.assertIsInstance(ret, GridXY)

    def test_get_mask(self):
        grid = self.get_gridxy()
        self.assertTrue(grid.is_vectorized)
        mask = grid.get_mask(create=True)
        self.assertEqual(mask.ndim, 2)
        self.assertFalse(np.any(mask))
        self.assertTrue(grid.is_vectorized)

        grid = self.get_gridxy()
        self.assertIsNone(grid.get_mask())

    def test_get_intersects(self):
        subset = box(100.7, 39.71, 102.30, 42.30)
        desired_manual = [[[40.0, 40.0], [41.0, 41.0], [42.0, 42.0]],
                          [[101.0, 102.0], [101.0, 102.0], [101.0, 102.0]]]
        desired_manual = np.array(desired_manual)

        grid = self.get_gridxy(crs=WGS84())
        # self.write_fiona_htmp(grid, 'grid')
        # self.write_fiona_htmp(GeometryVariable(value=subset), 'subset')
        sub, sub_slc = grid.get_intersects(subset, return_slice=True)

        self.assertFalse(sub.has_allocated_point)
        self.assertFalse(sub.has_allocated_polygon)
        self.assertFalse(sub.has_allocated_abstraction_geometry)
        # self.write_fiona_htmp(sub, 'sub')
        self.assertEqual(sub_slc, (slice(0, 3, None), slice(0, 2, None)))
        self.assertNumpyAll(sub.get_value_stacked(), desired_manual)
        point = sub.get_point()
        self.assertEqual(point.crs, grid.crs)

        # Test masks are updated.
        grid = self.get_gridxy(with_xy_bounds=True, with_parent=True)
        for t in ['xbounds', 'ybounds']:
            self.assertIn(t, grid.parent)
        subset = 'Polygon ((100.81193771626298883 42.17577854671281301, 101.13166089965399408 42.21211072664360842, 101.34965397923876651 41.18754325259516236, 103.68944636678200766 41.34013840830451159, 103.63858131487890546 41.22387543252595776, 100.77560553633219342 41.08581314878893664, 100.81193771626298883 42.17577854671281301))'
        subset = wkt.loads(subset)
        sub = grid.get_intersects(subset, cascade=True)
        self.assertTrue(sub.get_mask().any())
        self.assertTrue(sub.get_abstraction_geometry().get_mask().any())
        mask_slice = {'ydim': slice(1, 2), 'xdim': slice(1, 3)}

        sub_member_variables = get_variable_names(sub.get_member_variables())
        for v in sub.parent.values():
            if v.name in sub_member_variables and not isinstance(v,
                                                                 GeometryVariable) and v.name != sub.mask_variable.name:
                self.assertIsNone(v.get_mask())
            else:
                self.assertTrue(v.get_mask().any())
                self.assertFalse(v.get_mask().all())
                actual = v[mask_slice].get_mask()
                self.assertTrue(np.all(actual))

    def test_get_intersects_barbed_geometry(self):
        subset1 = 'Polygon ((100.79558316115701189 41.18854700413223213, 100.79558316115701189 40.80036157024792942, 102.13212035123964938 40.82493026859503971, 102.27953254132229688 41.47354390495867449, 103.62589721074378701 41.55707747933884377, 102.28444628099171609 41.66517975206611624, 102.06332799586775195 42.18112241735536827, 101.72919369834708903 42.13198502066115481, 101.72919369834708903 41.2671668388429751, 100.79558316115701189 41.18854700413223213))'
        subset1 = wkt.loads(subset1)
        subset2 = 'Polygon ((102.82100076148078927 43.17027003280225017, 103.26183018978443329 43.19503573102155514, 103.47481519447048015 42.70467490627928697, 101.38459026476100178 42.53131501874414511, 102.81604762183692969 42.86317537488285012, 102.82100076148078927 43.17027003280225017))'
        subset2 = wkt.loads(subset2)

        ################################################################################################################

        # Test with a single polygon.
        subset = subset1

        grid = self.get_gridxy()
        self.assertEqual(grid.abstraction, 'point')
        # self.write_fiona_htmp(grid, 'grid')
        # self.write_fiona_htmp(GeometryVariable(value=subset), 'subset')
        res = grid.get_intersects(subset, return_slice=True)

        grid_sub, slc = res

        mask_grid_sub = grid_sub.get_mask()
        self.assertEqual(grid_sub.shape, (2, 2))
        # self.log.debug(mask_grid_sub)
        self.assertTrue(np.any(mask_grid_sub))
        self.assertTrue(mask_grid_sub[1, 0])
        self.assertEqual(mask_grid_sub.sum(), 1)

        ################################################################################################################

        # Test with a multi-polygon.

        subset = MultiPolygon([subset1, subset2])
        grid = self.get_gridxy()

        res = grid.get_intersects(subset, return_slice=True)

        grid_sub, slc = res
        mask_grid_sub = grid_sub.get_mask()
        desired_mask = np.array([[False, False, True], [True, False, True], [True, True, False]])
        self.assertNumpyAll(mask_grid_sub, desired_mask)

    def test_get_intersects_bounds_sequence(self):
        keywords = dict(bounds=[True, False], use_bounds=[True, False])

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            # self.log.debug([ctr, k])
            # if ctr != 1: continue
            # Bounds may not be used if they are not present.
            if k.use_bounds and not k.bounds:
                continue

            y = self.get_variable_y(bounds=k.bounds)
            x = self.get_variable_x(bounds=k.bounds)
            grid = GridXY(x, y)

            bounds_sequence = box(-99, 39, -98, 39)
            bg = grid.get_intersects(bounds_sequence)
            self.assertNotEqual(grid.shape, bg.shape)
            self.assertTrue(bg.is_vectorized)

            with self.assertRaises(EmptySubsetError):
                bounds_sequence = box(1000, 1000, 1001, 10001)
                grid.get_intersects(bounds_sequence, use_bounds=k.use_bounds)

            bounds_sequence = box(-99999, 1, 1, 1000)
            bg2 = grid.get_intersects(bounds_sequence, use_bounds=k.use_bounds)

            for target in ['x', 'y']:
                original = getattr(grid, target).value
                sub = getattr(bg2, target).value
                self.assertNumpyAll(original, sub)

        # Test mask is not shared with subsetted grid.
        grid = self.get_gridxy()
        new_mask = grid.get_mask(create=True)
        new_mask[:, 1] = True
        grid.set_mask(new_mask)
        self.assertFalse(grid.has_bounds)

        bounds_sequence = box(101.5, 40.5, 103.5, 42.5)
        sub = grid.get_intersects(bounds_sequence, use_bounds=False)
        new_mask = sub.get_mask()
        new_mask.fill(True)
        sub.set_mask(new_mask)
        self.assertEqual(grid.get_mask().sum(), 4)

    @attr('mpi')
    def test_get_intersects_ordering(self):
        """Test grid ordering/origins do not influence grid subsetting."""

        keywords = {KeywordArguments.OPTIMIZED_BBOX_SUBSET: [False, True],
                    'should_wrap': [False, True],
                    'reverse_x': [False, True],
                    'reverse_y': [False, True],
                    'should_expand': [False, True],
                    }

        x_value = np.array([155., 160., 165., 170., 175., 180., 185., 190., 195., 200., 205.])
        y_value = np.array([-20., -15., -10., -5., 0., 5., 10., 15., 20.])
        bbox = [168., -12., 191., 5.3]

        for k in self.iter_product_keywords(keywords, as_namedtuple=False):
            reverse_x = k.pop('reverse_x')
            reverse_y = k.pop('reverse_y')
            should_expand = k.pop('should_expand')
            should_wrap = k.pop('should_wrap')

            ompi = OcgMpi()
            dx = ompi.create_dimension('dx', len(x_value), dist=True)
            dy = ompi.create_dimension('dy', len(y_value))
            ompi.update_dimension_bounds()

            if reverse_x:
                new_x_value = x_value.copy()
                new_x_value = np.flipud(new_x_value)
            else:
                new_x_value = x_value

            if reverse_y:
                new_y_value = y_value.copy()
                new_y_value = np.flipud(new_y_value)
            else:
                new_y_value = y_value

            if MPI_RANK == 0:
                x = Variable('x', new_x_value, 'dx')
                y = Variable('y', new_y_value, 'dy')
            else:
                x, y = [None, None]

            x, _ = variable_scatter(x, ompi)
            y, _ = variable_scatter(y, ompi)
            grid = GridXY(x, y, crs=Spherical())

            if should_expand:
                expand_grid(grid)

            if should_wrap:
                grid = deepcopy(grid)
                grid.wrap()
                actual_bbox = MultiPolygon([box(-180, -12, -169, 5.3), box(168, -12, 180, 5.3)])
            else:
                actual_bbox = box(*bbox)

            sub = grid.get_intersects(actual_bbox, **k)

            if should_wrap and not sub.is_empty:
                current_x_value = sub.x.value
                current_x_value[sub.x.value < 0] += 360

            self.assertEqual(sub.extent_global, (170.0, -10.0, 190.0, 5.0))

            if should_expand:
                desired = False
            else:
                desired = True
            self.assertEqual(grid.is_vectorized, desired)
            self.assertEqual(sub.is_vectorized, desired)

            self.assertFalse(grid.has_allocated_point)
            self.assertFalse(grid.has_allocated_polygon)

    @attr('mpi')
    def test_get_intersects_parallel(self):
        grid = self.get_gridxy()
        # Test with an empty subset.
        subset_geom = box(1000., 1000., 1100., 1100.)
        with self.assertRaises(EmptySubsetError):
            grid.get_intersects(subset_geom)

        # Test combinations.
        subset_geom = box(101.5, 40.5, 102.5, 42.)

        keywords = dict(is_vectorized=[True, False], has_bounds=[False, True], use_bounds=[False, True],
                        keep_touches=[True, False])

        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            grid = self.get_gridxy()
            if k.has_bounds:
                grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
                self.assertTrue(grid.has_bounds)

            # Cannot use bounds with a point grid abstraction.
            if k.use_bounds and grid.abstraction == 'point':
                continue

            grid_sub, slc = grid.get_intersects(subset_geom, keep_touches=k.keep_touches, use_bounds=k.use_bounds,
                                                return_slice=True)

            if k.has_bounds:
                self.assertTrue(grid.has_bounds)

            # Test geometries are filled appropriately after allocation.
            if not grid_sub.is_empty:
                for t in grid_sub.get_abstraction_geometry().value.flat:
                    self.assertIsInstance(t, BaseGeometry)
            self.assertIsInstance(grid_sub, GridXY)
            if k.keep_touches:
                if k.has_bounds and k.use_bounds:
                    desired = (slice(0, 3, None), slice(0, 3, None))
                else:
                    desired = (slice(1, 3, None), slice(1, 2, None))
            else:
                if k.has_bounds and k.use_bounds:
                    desired = (slice(1, 3, None), slice(1, 2, None))
                else:
                    desired = (slice(1, 2, None), slice(1, 2, None))
            if not grid.is_empty:
                self.assertEqual(grid.has_bounds, k.has_bounds)
                self.assertTrue(grid.is_vectorized)
            self.assertEqual(slc, desired)

        # Test against a file. #########################################################################################
        subset_geom = box(101.5, 40.5, 102.5, 42.)

        if MPI_RANK == 0:
            path_grid = self.get_temporary_file_path('grid.nc')
        else:
            path_grid = None
        path_grid = MPI_COMM.bcast(path_grid)

        grid_to_write = self.get_gridxy()
        field = OcgField(grid=grid_to_write)
        field.write(path_grid, driver=DriverNetcdfCF)
        MPI_COMM.Barrier()

        rd = RequestDataset(uri=path_grid)
        x = SourcedVariable(name='x', request_dataset=rd)
        self.assertIsNone(x._value)
        y = SourcedVariable(name='y', request_dataset=rd)
        self.assertIsNone(x._value)
        self.assertIsNone(y._value)

        grid = GridXY(x, y)

        for target in [grid._y_name, grid._x_name]:
            self.assertIsNone(grid.parent[target]._value)
        self.assertTrue(grid.is_vectorized)

        sub, slc = grid.get_intersects(subset_geom, comm=MPI_COMM, return_slice=True)

        self.assertEqual(slc, (slice(1, 3, None), slice(1, 2, None)))
        self.assertIsInstance(sub, GridXY)

        # The file may be deleted before other ranks open.
        MPI_COMM.Barrier()

    def test_get_intersects_no_slice(self):
        """Test an intersects operations with no slice."""

        x = Variable(name='x', value=[1, 2, 3, 4, 5], dtype=float, dimensions='x')
        y = Variable(name='y', value=[1, 2, 3, 4, 5, 6, 7], dtype=float, dimensions='y')
        grid = GridXY(x, y)
        subset_geom = Point(3, 4)
        sub, the_slice = grid.get_intersects(subset_geom, return_slice=True, apply_slice=False)
        self.assertEqual(np.sum(np.invert(sub.get_mask())), 1)
        self.assertEqual(grid.shape, sub.shape)
        self.assertEqual(the_slice, (slice(3, 4, None), slice(2, 3, None)))
        sub2 = sub[the_slice]
        self.assertEqual(subset_geom, sub2.get_point().get_value().flatten()[0])

    def test_get_intersects_small(self):
        """Test with a subset inside of one of the cells."""

        subset = 'MULTIPOLYGON (((-71.79019426324761 41.60130736620898, -71.79260526324985 41.64175836624665, -71.7882492632458 41.72160336632101, -71.79783126325472 42.00427436658427, -71.49743026297496 42.00925336658891, -71.37864426286433 42.01371336659307, -71.38240526286783 41.97926336656098, -71.38395326286928 41.88843936647639, -71.33308626282189 41.89603136648346, -71.34249326283066 41.8757833664646, -71.33454226282325 41.85790336644796, -71.34548326283345 41.81316136640628, -71.33979826282815 41.78442536637952, -71.31932826280908 41.77219536636813, -71.26662826276001 41.74974336634722, -71.22897626272494 41.70769436630806, -71.28400126277619 41.67954936628185, -71.36738726285384 41.7413503663394, -71.39358026287823 41.76115536635785, -71.36901226285535 41.70329136630396, -71.41924726290215 41.65221236625639, -71.42731826290965 41.48668936610223, -71.48988826296792 41.39208536601413, -71.72226426318434 41.32726436595375, -71.86667826331885 41.32276936594957, -71.84777226330124 41.32534836595197, -71.83686926329108 41.34196136596745, -71.84599526329959 41.40385436602509, -71.8027432632593 41.41582936603623, -71.79019426324761 41.60130736620898)), ((-71.19880826269684 41.67850036628087, -71.14121226264319 41.65527336625924, -71.11713226262077 41.49306236610817, -71.19993726269789 41.46331836608047, -71.19880826269684 41.67850036628087)), ((-71.26916926276238 41.62126836622758, -71.21944726271606 41.63564236624096, -71.23867326273397 41.47484936609121, -71.28800726277991 41.48361936609938, -71.3495252628372 41.4458573660642, -71.26916926276238 41.62126836622758)))'
        subset = wkt.loads(subset)

        grid = self.get_gridxy_global(resolution=5.0)

        sub, slc = grid.get_intersects(subset, return_slice=True)
        self.assertEqual(sub.shape, (1, 1))
        self.assertEqual(sub.get_mask().sum(), 0)

    @attr('mpi')
    def test_get_intersection_state_boundaries(self):
        path_shp = self.path_state_boundaries
        geoms = []
        with fiona.open(path_shp) as source:
            for record in source:
                geom = shape(record['geometry'])
                geoms.append(geom)

        gvar = GeometryVariable(value=geoms, dimensions='ngeom')
        gvar_sub = gvar.get_unioned()

        if gvar_sub is not None:
            subset = gvar_sub.get_value().flatten()[0]

        else:
            subset = None
        subset = MPI_COMM.bcast(subset)
        resolution = 2.0

        keywords = dict(with_bounds=[False])

        for k in self.iter_product_keywords(keywords):
            grid = self.get_gridxy_global(resolution=resolution, with_bounds=k.with_bounds)
            res = grid.get_intersection(subset)

            if not res.is_empty:
                self.assertTrue(res.get_mask().any())
            else:
                self.assertIsInstance(res, GeometryVariable)

            if k.with_bounds:
                area = res.area
                if area is None:
                    area = 0.0
                else:
                    area = area.sum()
                areas = MPI_COMM.gather(area)
                if MPI_RANK == 0:
                    area_global = sum(areas)
                    self.assertAlmostEqual(area_global, 1096.0819224080542)
            else:
                mask = res.get_mask()
                if mask is None:
                    masked = 0
                else:
                    masked = mask.sum()
                masked = MPI_COMM.gather(masked)
                if MPI_RANK == 0:
                    total_masked = sum(masked)
                    self.assertEqual(total_masked, 858)

    @attr('mpi')
    def test_get_intersects_state_boundaries(self):
        path_shp = self.path_state_boundaries
        # path_shp = '/home/benkoziol/Dropbox/NESII/project/ocg/bin/shp/world_countries/world_countries.shp'
        geoms = []
        with fiona.open(path_shp) as source:
            for record in source:
                geom = shape(record['geometry'])
                geoms.append(geom)

        gvar = GeometryVariable(value=geoms, dimensions='ngeom')
        gvar_sub = gvar.get_unioned()

        if gvar_sub is not None:
            subset = gvar_sub.get_value().flatten()[0]

        else:
            subset = None
        subset = MPI_COMM.bcast(subset)
        resolution = 1.0

        # for with_bounds in [True]:
        for with_bounds in [False, True]:
            grid = self.get_gridxy_global(resolution=resolution, with_bounds=with_bounds)
            # barrier_print('before get_intersects:', with_bounds)
            # t1 = time.time()
            res = grid.get_intersects(subset, return_slice=True)
            # t2 = time.time()
            # barrier_print(t2 - t1)
            # continue
            grid_sub, slc = res

            mask = Variable('mask_after_subset', grid_sub.get_mask(),
                            dimensions=grid_sub.dimensions, is_empty=grid_sub.is_empty)
            mask = variable_gather(mask)

            if MPI_RANK == 0:
                mask_sum = np.invert(mask.value).sum()
                mask_shape = mask.shape
            else:
                mask_sum = None
                mask_shape = None
            mask_sum = MPI_COMM.bcast(mask_sum)
            mask_shape = MPI_COMM.bcast(mask_shape)

            if with_bounds:
                self.assertEqual(mask_shape, (54, 113))
                self.assertEqual(slc, (slice(108, 162, None), slice(1, 114, None)))
                self.assertEqual(mask_sum, 1358)
            else:
                if MPI_SIZE == 2:
                    grid_bounds_global = [dim.bounds_global for dim in grid_sub.dimensions]
                    self.assertEqual(grid_bounds_global, [(0, 52), (0, 105)])
                self.assertEqual(mask_shape, (52, 105))
                self.assertEqual(slc, (slice(109, 161, None), slice(8, 113, None)))
                self.assertEqual(mask_sum, 1087)

            if MPI_RANK == 0:
                path = self.get_temporary_file_path('foo.nc')
            else:
                path = None
            path = MPI_COMM.bcast(path)
            field = OcgField(grid=grid_sub)
            field.write(path)
            MPI_COMM.Barrier()

    def test_get_value_polygons(self):
        """Test ordering of vertices when creating from corners is slightly different."""

        keywords = dict(with_bounds=[False, True])
        for k in self.iter_product_keywords(keywords, as_namedtuple=True):
            grid = self.get_polygon_array_grid(with_bounds=k.with_bounds)
            self.assertTrue(grid.is_vectorized)
            if not k.with_bounds:
                grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
            desired = self.polygon_value
            for ageom, gridgeom in zip(desired.flat, grid.get_abstraction_geometry().get_value().flat):
                self.assertEqual(ageom.bounds, gridgeom.bounds)
                self.assertEqual(ageom.area, gridgeom.area)
                self.assertEqual(type(ageom), type(gridgeom))
                self.assertEqual(ageom.centroid, gridgeom.centroid)
                self.assertAsSetEqual(np.array(ageom.exterior).flatten().tolist(),
                                      np.array(gridgeom.exterior).flatten().tolist())
            self.assertTrue(grid.is_vectorized)

    def test_reorder(self):
        x = np.linspace(1, 360, num=7, dtype=float)
        x = Variable('lon', value=x, dimensions='dimx')
        y = Variable('lat', value=[-40, -20, 0, 20, 40], dimensions='dimy')
        t = Variable('time', value=[1, 2, 3], dimensions='dimt')

        data = Variable(name='data', value=np.zeros((3, 5, 7)),
                        dimensions=['dimt', 'dimy', 'dimx'], dtype=float)
        data.value[:] = 10.
        data.value[:, :, 3:x.shape[0]] = 20.
        data_mask = data.get_mask(create=True)
        data_mask[:, :, 3:x.shape[0]] = 1
        data.set_mask(data_mask)

        parent = VariableCollection(variables=[x, y, t, data])

        grid = GridXY(parent['lon'], parent['lat'], crs=Spherical(), parent=parent)
        desired_y = grid.y.value.copy()

        grid.wrap()

        self.assertFalse(np.any(data[:, :, 0:3].value == 20.))
        self.assertFalse(np.any(data.get_mask()[:, :, 0:3]))

        grid.reorder()
        actual = grid.x.value.tolist()
        desired = [-179.5, -119.66666666666666, -59.833333333333314, 0.0, 1.0, 60.833333333333336, 120.66666666666667]
        self.assertEqual(actual, desired)

        self.assertNumpyAll(desired_y, grid.y.value)
        rdata = grid.parent['data']
        self.assertTrue(np.all(rdata[:, :, 0:3].value == 20.))
        self.assertTrue(np.all(rdata.get_mask()[:, :, 0:4]))
        self.assertFalse(np.any(rdata.get_mask()[:, :, 4:]))
        self.assertNumpyMayShareMemory(rdata.value, data.value)

    def test_resolution(self):
        for grid in self.get_iter_gridxy():
            self.assertEqual(grid.resolution, 1.)

    def test_set_extrapolated_bounds(self):
        value_grid = [[[40.0, 40.0, 40.0, 40.0], [39.0, 39.0, 39.0, 39.0], [38.0, 38.0, 38.0, 38.0]],
                      [[-100.0, -99.0, -98.0, -97.0], [-100.0, -99.0, -98.0, -97.0], [-100.0, -99.0, -98.0, -97.0]]]
        actual_corners = [
            [[[40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5], [40.5, 40.5, 39.5, 39.5]],
             [[39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5], [39.5, 39.5, 38.5, 38.5]],
             [[38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5], [38.5, 38.5, 37.5, 37.5]]],
            [[[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]],
             [[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]],
             [[-100.5, -99.5, -99.5, -100.5], [-99.5, -98.5, -98.5, -99.5], [-98.5, -97.5, -97.5, -98.5],
              [-97.5, -96.5, -96.5, -97.5]]]]

        for should_extrapolate in [False, True]:
            y = Variable(name='y', value=value_grid[0], dimensions=['ydim', 'xdim'])
            x = Variable(name='x', value=value_grid[1], dimensions=['ydim', 'xdim'])
            if should_extrapolate:
                y.set_extrapolated_bounds('ybounds', 'bounds')
                x.set_extrapolated_bounds('xbounds', 'bounds')
            grid = GridXY(x, y)
            try:
                grid.set_extrapolated_bounds('ybounds', 'xbounds', 'bounds')
            except BoundsAlreadyAvailableError:
                self.assertTrue(should_extrapolate)
            else:
                np.testing.assert_equal(grid.y.bounds.value, actual_corners[0])
                np.testing.assert_equal(grid.x.bounds.value, actual_corners[1])

        # Test vectorized.
        y = Variable(name='y', value=[1., 2., 3.], dimensions='yy')
        x = Variable(name='x', value=[10., 20., 30.], dimensions='xx')
        grid = GridXY(x, y)
        grid.set_extrapolated_bounds('ybounds', 'xbounds', 'bounds')
        self.assertEqual(grid.x.bounds.ndim, 2)
        self.assertTrue(grid.is_vectorized)

    def test_set_extrapolated_bounds_empty(self):
        """Test bounds extrapolation works on empty grids. Grid expansion for empty variables is also tested."""

        dimx = Dimension('x', 2, is_empty=True, dist=True)
        x = Variable('x', dimensions=dimx)
        y = Variable('3', dimensions=Dimension('y', 3))
        grid = GridXY(x, y)

        self.assertTrue(dimx.is_empty)
        self.assertTrue(x.is_empty)
        self.assertTrue(grid.is_empty)
        self.assertFalse(grid.has_bounds)
        self.assertEqual(grid.abstraction, 'point')

        grid.set_extrapolated_bounds('xbnds', 'ybnds', 'bounds')

        self.assertTrue(grid.has_bounds)
        self.assertTrue(grid.is_empty)
        self.assertEqual(grid.abstraction, 'polygon')

    def test_setitem(self):
        grid = self.get_gridxy()
        self.assertNotIn('point', grid.parent)
        self.assertFalse(np.any(grid.get_mask()))
        grid2 = deepcopy(grid)
        grid2.x[:] = Variable(value=111, name='scalar111', dimensions=[])
        grid2.y[:] = Variable(value=222, name='scalar222', dimensions=[])
        grid2.set_mask(np.ones(grid2.shape))
        self.assertTrue(grid2.get_mask().all())
        grid[:, :] = grid2
        self.assertTrue(np.all(grid.get_mask()))
        self.assertEqual(grid.x.value.mean(), 111)
        self.assertEqual(grid.y.value.mean(), 222)

    def test_set_mask(self):
        grid = self.get_gridxy()
        grid.parent['coordinate_system'] = Variable(name='coordinate_system')
        self.assertFalse(np.any(grid.get_mask()))
        mask = np.zeros(grid.shape, dtype=bool)
        mask[1, 1] = True
        self.assertTrue(grid.is_vectorized)
        grid.set_mask(mask)
        self.assertTrue(np.all(grid.get_mask()[1, 1]))
        self.assertIn('coordinate_system', grid.parent)
        self.assertTrue(grid.is_vectorized)
        for mvar in grid.get_member_variables():
            if mvar.name != grid._mask_name:
                self.assertIsNone(mvar.get_mask())

        path = self.get_temporary_file_path('foo.nc')
        grid.write(path)
        # self.ncdump(path)
        nvc = VariableCollection.read(path)
        ngrid = GridXY(nvc['x'], nvc['y'], parent=nvc)
        # Mask is not written to coordinate variables.
        for mvar in ngrid.get_member_variables():
            if mvar.name == grid.mask_variable.name:
                self.assertTrue(mvar.get_mask()[1, 1])
            else:
                self.assertIsNone(mvar.get_mask())

        # Test with a parent.
        grid = self.get_gridxy(with_parent=True)
        for k in ['tas', 'rhs']:
            self.assertIsNone(grid.parent[k].get_mask())
        new_mask = grid.get_mask(create=True)
        self.assertFalse(new_mask.any())
        new_mask[1:3, 1] = True
        grid.set_mask(new_mask, cascade=True)
        for k in ['tas', 'rhs']:
            backref_var = grid.parent[k]
            mask = backref_var.get_mask()
            self.assertTrue(mask.any())
            if k == 'tas':
                self.assertTrue(mask[:, 1, 1:3].all())
            if k == 'rhs':
                self.assertTrue(mask[1:3, 1, :].all())
            self.assertEqual(mask.sum(), 20)

    def test_shape(self):
        for grid in self.get_iter_gridxy():
            self.assertEqual(grid.shape, (4, 3))
            self.assertEqual(grid.ndim, 2)

    def test_update_crs(self):
        grid = self.get_gridxy(crs=WGS84())
        grid.set_extrapolated_bounds('xbounds', 'ybounds', 'bounds')
        self.assertIsNotNone(grid.y.bounds)
        self.assertIsNotNone(grid.x.bounds)
        to_crs = CoordinateReferenceSystem(epsg=3395)
        grid.update_crs(to_crs)
        self.assertEqual(grid.crs, to_crs)
        for element in [grid.x, grid.y]:
            for target in [element.value, element.bounds.value]:
                self.assertTrue(np.all(target > 10000))
