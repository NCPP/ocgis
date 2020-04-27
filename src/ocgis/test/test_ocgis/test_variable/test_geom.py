import itertools
import os
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import shapely
from mock import mock
from nose.plugins.skip import SkipTest
from numpy.ma import MaskedArray
from shapely import wkt
from shapely.geometry import Point, box, MultiPoint, LineString, Polygon
from shapely.geometry.base import BaseMultipartGeometry
from shapely.geometry.multilinestring import MultiLineString
from shapely.geometry.multipolygon import MultiPolygon

import ocgis
from ocgis import RequestDataset, vm, Field, GeomCabinetIterator
from ocgis import env, CoordinateReferenceSystem
from ocgis.constants import DMK, WrappedState, OcgisConvention, DriverKey
from ocgis.exc import EmptySubsetError, NoInteriorsError, RequestableFeature
from ocgis.spatial.grid import Grid, get_geometry_variable
from ocgis.test import strings
from ocgis.test.base import attr, AbstractTestInterface, create_gridxy_global, TestBase
from ocgis.variable.base import Variable, VariableCollection
from ocgis.variable.crs import WGS84, Spherical, Cartesian
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable, GeometryProcessor, get_split_polygon_by_node_threshold, \
    GeometrySplitter, do_remove_self_intersects_multi
from ocgis.vmachine.mpi import OcgDist, MPI_RANK, variable_scatter, MPI_SIZE, variable_gather, MPI_COMM


class FixtureSelfIntersectingPolygon(object):

    @property
    def fixture_self_intersecting_polygon_coords(self):
        coords = [(0, 0), (0, 3), (2, 3), (2, 2), (1, 2), (1, 1), (2, 1), (2, 2), (3, 2), (3, 0), (0, 0)]
        return coords

    @property
    def fixture_self_intersecting_polygon_coords_first(self):
        # Fixture has the first coordinate as the repeating one.
        coords = [(2, 2), (3, 2), (3, 0), (0, 0), (0, 3), (2, 3), (2, 2), (1, 2), (1, 1), (2, 1), (2, 2)]
        return coords


class TestGeom(TestBase, FixtureSelfIntersectingPolygon):

    def run_do_remove_self_intersects(self, fixture, debug=False):
        poly = Polygon(fixture)
        if debug:
            gvar = GeometryVariable.from_shapely(poly)
            gvar.write_vector('/tmp/vector.shp')

        new_poly = do_remove_self_intersects_multi(poly)
        self.assertEqual(np.array(poly.exterior.coords).shape[0] - 1, np.array(new_poly.exterior.coords).shape[0])
        self.assertTrue(new_poly.is_valid)
        if debug:
            GeometryVariable.from_shapely(new_poly).write_vector('/tmp/new_vector.shp')

    def test_do_remove_self_intersects(self):
        self.run_do_remove_self_intersects(self.fixture_self_intersecting_polygon_coords, debug=False)

    def test_do_remove_self_intersects_first(self):
        self.run_do_remove_self_intersects(self.fixture_self_intersecting_polygon_coords_first, debug=False)

    def test_dev(self):
        #tdk:rm
        self.skipTest("dev test")
        path = '/home/benkoziol/htmp/bad_geoms/bad_geom.shp'
        new_records = []
        gci = GeomCabinetIterator(path=path)
        n = len(gci)
        for ctr, row in enumerate(gci):
            print('{} of {}'.format(ctr, n))
            print(row)
            poly = row['geom']
            new_poly = do_remove_self_intersects_multi(poly)
            self.assertTrue(new_poly.is_valid)
            row['geom'] = new_poly
            new_records.append(row)
        field = Field.from_records(new_records)
        field.write(os.path.expanduser('~/htmp/clean_geoms.shp'), driver='vector')


class TestGeometryProcessor(AbstractTestInterface):
    def test_iter_intersection(self):

        def the_geometry_iterator():
            to_yield = [None] * 2
            to_yield[0] = box(1.0, 2.0, 3.0, 4.0)
            to_yield[1] = box(10.0, 20.0, 30.0, 40.0)
            for idx, ty in enumerate(to_yield):
                yield idx, ty

        subset_geometry = box(1.5, 2.5, 2.5, 3.5)

        gp = GeometryProcessor(the_geometry_iterator(), subset_geometry)

        all_actual = list(gp.iter_intersection())

        actual = all_actual[0]
        self.assertEqual(actual[0], 0)
        self.assertEqual(actual[1], True)
        self.assertEqual(actual[2].bounds, (1.5, 2.5, 2.5, 3.5))

        actual = all_actual[1]
        self.assertEqual(actual[0], 1)
        self.assertEqual(actual[1], False)
        self.assertEqual(actual[2], None)

    def test_iter_intersects(self):

        desired = {False: [(0, False), (1, False), (2, False), (3, True), (4, False)],
                   True: [(0, False), (1, True), (2, False), (3, True), (4, False)]}

        def the_geometry_iterator():
            x = [1, 2, 3, 4, 5]
            y = [6, 7, 8, 9, 10]
            mask = [False, False, True, False, False, False]
            for idx in range(len(x)):
                if mask[idx]:
                    yld = None
                else:
                    yld = Point(x[idx], y[idx])
                yield idx, yld

        subset_geometry = box(2.0, 7.0, 4.5, 9.5)

        for keep_touches in [False, True]:
            gp = GeometryProcessor(the_geometry_iterator(), subset_geometry, keep_touches=keep_touches)
            actual = list(gp.iter_intersects())
            for idx_actual in range(len(actual)):
                self.assertEqual(actual[idx_actual][0:2], desired[keep_touches][idx_actual])
                if idx_actual == 2:
                    self.assertIsNone(actual[idx_actual][2])
                else:
                    self.assertIsInstance(actual[idx_actual][2], Point)

        # Test the object may only be used once.
        with self.assertRaises(ValueError):
            list(gp.iter_intersects())


class FixturePolygonWithHole(object):
    @property
    def fixture_polygon_with_hole(self):
        outer_box = box(2.0, 10.0, 4.0, 20.0)
        inner_box = box(2.5, 10.5, 3.5, 15.5)

        outer_coords = list(outer_box.exterior.coords)
        inner_coords = [list(inner_box.exterior.coords)]

        with_interior = Polygon(outer_coords, holes=inner_coords)

        return with_interior


class TestGeometrySplitter(TestBase, FixturePolygonWithHole):
    def test_init(self):
        ge = GeometrySplitter(self.fixture_polygon_with_hole)
        self.assertIsInstance(ge.geometry, Polygon)

        # Test a geometry with no holes.
        with self.assertRaises(NoInteriorsError):
            GeometrySplitter(box(1, 2, 3, 4))

    def test_create_split_vector_dict(self):
        ge = GeometrySplitter(self.fixture_polygon_with_hole)
        desired = [{'rows': (9.999999, 13.0, 20.000001), 'cols': (1.999999, 3.0, 4.000001)}]
        actual = list([ge.create_split_vector_dict(i) for i in ge.iter_interiors()])
        self.assertEqual(actual, desired)

    def test_create_split_polygons(self):
        ge = GeometrySplitter(self.fixture_polygon_with_hole)
        spolygons = ge.create_split_polygons(list(ge.iter_interiors())[0])
        self.assertEqual(len(spolygons), 4)

        actual = [sp.bounds for sp in spolygons]
        desired = [(1.999999, 9.999999, 3.0, 13.0), (3.0, 9.999999, 4.000001, 13.0),
                   (3.0, 13.0, 4.000001, 20.000001), (1.999999, 13.0, 3.0, 20.000001)]
        self.assertEqual(actual, desired)

    def test_split(self):
        to_test = [self.fixture_polygon_with_hole,
                   MultiPolygon([self.fixture_polygon_with_hole, box(200, 100, 300, 400)])]
        desired_counts = {0: 4, 1: 5}

        for ctr, t in enumerate(to_test):
            ge = GeometrySplitter(t)
            split = ge.split()

            self.assertEqual(len(split), desired_counts[ctr])
            self.assertEqual(split.area, t.area)

            actual_bounds = [g.bounds for g in split]
            actual_areas = [g.area for g in split]

            desired_bounds = [(2.0, 10.0, 3.0, 13.0), (3.0, 10.0, 4.0, 13.0),
                              (3.0, 13.0, 4.0, 20.0), (2.0, 13.0, 3.0, 20.0)]
            desired_areas = [1.75, 1.75, 5.75, 5.75]

            if ctr == 1:
                desired_bounds.append((200.0, 100.0, 300.0, 400.0))
                desired_areas.append(30000.0)

            self.assertEqual(actual_bounds, desired_bounds)
            self.assertEqual(actual_areas, desired_areas)

    def test_iter_interiors(self):
        ge = GeometrySplitter(self.fixture_polygon_with_hole)
        actual = list([g.bounds for g in ge.iter_interiors()])
        self.assertEqual(actual, [(2.5, 10.5, 3.5, 15.5)])


class TestGeometryVariable(AbstractTestInterface, FixturePolygonWithHole, FixtureSelfIntersectingPolygon):

    @staticmethod
    def get_geometryvariable_with_parent():
        vpa = np.array([None, None, None])
        vpa[:] = [Point(1, 2), Point(3, 4), Point(5, 6)]
        value = np.arange(0, 30).reshape(10, 3)
        tas = Variable(name='tas', value=value, dimensions=['time', 'ngeom'])
        backref = Field(variables=[tas])
        pa = GeometryVariable(value=vpa, parent=backref, name='point', dimensions='ngeom')
        backref[pa.name] = pa
        return pa

    def test_init(self):
        # Test empty.
        gvar = GeometryVariable()
        self.assertEqual(gvar.dtype, object)

        gvar = self.get_geometryvariable()
        self.assertIsInstance(gvar.get_masked_value(), MaskedArray)
        self.assertEqual(gvar.ndim, 1)

        # The geometry variable should set itself as the representative geometry on its parent field if that parent does
        # not have a representative geometry set.
        self.assertIsNotNone(gvar.parent.geom)

        # Test with a parent that already has a geometry.
        field = Field()
        field.set_geom(GeometryVariable(name='empty'))
        gvar = self.get_geometryvariable(parent=field)
        self.assertEqual(field.geom.name, 'empty')
        self.assertIn(gvar.name, field)

        # Test passing a "crs".
        gvar = self.get_geometryvariable(crs=WGS84(), name='my_geom', dimensions='ngeom')
        self.assertEqual(gvar.crs, WGS84())

        # Test using lines.
        line1 = LineString([(0, 0), (1, 1)])
        line2 = LineString([(1, 1), (2, 2)])
        gvar = GeometryVariable(value=[line1, line2], dimensions='two')
        self.assertTrue(gvar.get_value()[1].almost_equals(line2))
        self.assertEqual(gvar.geom_type, line1.geom_type)
        lines = MultiLineString([line1, line2])
        lines2 = [lines, lines]
        for actual in [lines, lines2, lines]:
            gvar2 = GeometryVariable(value=actual, dimensions='ngeom')
            self.assertTrue(gvar2.get_value()[0].almost_equals(lines))
            self.assertEqual(gvar2.geom_type, lines.geom_type)
            self.assertTrue(gvar2.shape[0] > 0)
            self.assertIsNone(gvar2.get_mask())

    @attr('data', 'mpi')
    def test_system_spatial_averaging_from_file(self):
        rd_nc = self.test_data.get_rd('cancm4_tas')

        rd_shp = RequestDataset(self.path_state_boundaries)
        field_shp = rd_shp.get()

        actual = field_shp.dimension_map.get_variable(DMK.GEOM)
        self.assertIsNotNone(actual)
        actual = field_shp.dimension_map.get_dimension(DMK.GEOM)
        self.assertEqual(len(actual), 1)

        self.assertEqual(field_shp.crs, WGS84())

        try:
            index_geom = np.where(field_shp['STATE_NAME'].get_value() == 'Nebraska')[0][0]
        except IndexError:
            # Not found on rank.
            polygon_field = None
        else:
            polygon_field = field_shp.get_field_slice({'geom': index_geom})
        polygon_field = MPI_COMM.gather(polygon_field)
        if MPI_RANK == 0:
            for p in polygon_field:
                if p is not None:
                    polygon_field = p
                    break
        polygon_field = MPI_COMM.bcast(polygon_field)
        polygon_field.unwrap()
        polygon = polygon_field.geom.get_value()[0]

        field_nc = rd_nc.get()
        sub_field_nc = field_nc.get_field_slice({'time': slice(0, 10)})
        self.assertEqual(sub_field_nc['tas']._dimensions, field_nc['tas']._dimensions)
        sub = sub_field_nc.grid.get_intersects(polygon)

        # When split across two processes, there are floating point summing differences.
        desired = {1: 2734.5195, 2: 2740.4014}
        with vm.scoped_by_emptyable('grid intersects', sub):
            if not vm.is_null:
                abstraction_geometry = sub.get_abstraction_geometry()
                sub.parent.add_variable(abstraction_geometry, force=True)
                unioned = abstraction_geometry.get_unioned(spatial_average='tas')
                if unioned is not None:
                    tas = unioned.parent['tas']
                    self.assertFalse(tas.is_empty)
                    self.assertAlmostEqual(tas.get_value().sum(), desired[vm.size], places=4)

    def test_system_spatial_averaging_weights(self):
        """Test creating averaging weights from first principles."""

        # x/y coordinate arrays and grid objects. Coordinates may be two-dimensional otherwise,
        x = ocgis.Variable(name='xc', dimensions='dimx', value=np.arange(5, 360, 10, dtype=float))
        y = ocgis.Variable(name='yc', dimensions='dimy', value=np.arange(-85, 90, 10, dtype=float))
        grid = ocgis.Grid(x, y, crs=ocgis.crs.Spherical())
        # Create spatial bounds on the grid coordinates. This allows us to use polygons as opposed to points for the
        # spatial averaging.
        grid.set_extrapolated_bounds('xc_bounds', 'yc_bounds', 'bounds')
        self.assertEqual(grid.abstraction, ocgis.constants.Topology.POLYGON)

        # This is the subset geometry. OCGIS geometry variables may be used to take advantage of wrapping and coordinate
        # system conversion.
        subset_geom = shapely.geometry.box(52, -70, 83, 10)

        # Perform an intersection. First, data is reduced in spatial extent using an intersects operation. A
        # clip/intersection is then performed for each geometry object.
        sub, slc = grid.get_intersection(subset_geom, return_slice=True)
        slc = {dim.name: se for dim, se in zip(grid.dimensions, slc)}  # Just how to convert to a dictionary slice...
        # Weights are computed on demand and is equal to original_area/clipped_area.
        weights = sub.weights
        self.assertAlmostEqual(weights.sum(), 24.799999999999997)

        # The OCGIS operation "get_unioned" will apply weights and union associated geometries. Works in parallel...
        # u = sub.get_unioned(spatial_average=[<varnames>, ...])

    def test_area(self):
        gvar = self.get_geometryvariable()
        self.assertTrue(np.all(gvar.area == 0))

    def test_as_shapely(self):
        coords = (12, 3, 15, 4)
        bbox = box(*coords)
        gvar = GeometryVariable(value=bbox, is_bbox=True, dimensions='geom', crs=Spherical())
        geom = gvar.as_shapely()
        self.assertEqual(geom.bounds, coords)

    def test_convert_to_geometry_coordinates_multipolygon(self):
        p1 = 'Polygon ((-116.94238466549290933 52.12861711455555991, -82.00526805089285176 61.59075286434307372, -59.92695130138864101 31.0207758265680269, -107.72286778108455962 22.0438778075388484, -122.76523743459291893 37.08624746104720771, -116.94238466549290933 52.12861711455555991))'
        p2 = 'Polygon ((-63.08099655131782413 21.31602121140134898, -42.70101185946779765 9.42769680782217279, -65.99242293586783603 9.912934538580501, -63.08099655131782413 21.31602121140134898))'
        p1 = wkt.loads(p1)
        p2 = wkt.loads(p2)

        mp1 = MultiPolygon([p1, p2])
        mp2 = mp1.buffer(0.1)
        geoms = [mp1, mp2]
        gvar = GeometryVariable(name='gc', value=geoms, dimensions='gd')

        # Test the element node connectivity arrays.
        results = []
        for pack in [False, True]:
            gc = gvar.convert_to(pack=pack)
            self.assertEqual(gc.dimension_map.get_driver(), DriverKey.NETCDF_UGRID)
            self.assertTrue(gc.has_multi)
            self.assertIn(OcgisConvention.Name.MULTI_BREAK_VALUE, gc.cindex.attrs)
            # Test multi-break values are part of the element node connectivity arrays.
            actual = gc.cindex.get_value()
            for idx, ii in enumerate(actual.flat):
                self.assertGreater(np.sum(ii == OcgisConvention.Value.MULTI_BREAK_VALUE), 0)
            results.append(actual)
            self.assertIsNotNone(gc.x.get_value())
            self.assertIsNotNone(gc.y.get_value())

            maxes = []
            for ii in actual.flat:
                maxes.append(ii.max())
            actual_max = max(maxes)
            for c in [gc.x, gc.y]:
                self.assertEqual(c.size - 1, actual_max)

            geoms = list(gc.iter_geometries())
            for ctr, g in enumerate(geoms):
                self.assertIsInstance(g[1], BaseMultipartGeometry)
            self.assertEqual(ctr, 1)

            self.assertPolygonSimilar(geoms[0][1], mp1)
            self.assertPolygonSimilar(geoms[1][1], mp2)

        for idx in range(len(results[0])):
            self.assertNumpyAll(results[0][idx], results[1][idx])

    def test_convert_to_geometry_coordinates_multipolygon_node_threshold(self):
        mp = wkt.loads(strings.S7)
        desired_count = len(get_split_polygon_by_node_threshold(mp, 10))
        self.assertGreater(desired_count, len(mp))
        gvar = GeometryVariable.from_shapely(mp)
        gc = gvar.convert_to(node_threshold=10)
        actual_count = gc.cindex.get_value()[0]
        actual_count = np.sum(actual_count == OcgisConvention.Value.MULTI_BREAK_VALUE) + 1
        self.assertEqual(actual_count, desired_count)

    def test_convert_to_geometry_coordinates_points(self):
        pt1 = Point(1, 2, 3)
        pt2 = Point(3, 4, 4)
        pt3 = Point(5, 6, 5)
        pt4 = Point(7, 8, 6)
        crs = WGS84()
        value = [pt1, pt2, pt3, pt4]
        mask = [True, False, True, False]

        gvar = GeometryVariable(value=value, dimensions='ngeom', crs=crs, mask=mask)

        gvar_mask = gvar.get_mask().tolist()
        self.assertEqual(mask, gvar_mask)
        actual = gvar.convert_to()
        self.assertEqual(actual.get_mask().tolist(), gvar_mask)

        for actual_geom, desired_geom in zip(actual.get_geometry_iterable(use_mask=False), value):
            self.assertEqual(actual_geom[1], desired_geom)

        self.assertEqual(actual.crs, crs)

    def test_convert_to_geometry_coordinates_polygon_interior(self):
        ph = self.fixture_polygon_with_hole
        gvar = GeometryVariable.from_shapely(ph)
        desired_count = len(GeometrySplitter(ph).split())

        keywords = dict(split_interiors=[True, False])
        for k in self.iter_product_keywords(keywords):
            try:
                gc = gvar.convert_to(split_interiors=k.split_interiors)
            except ValueError:
                self.assertFalse(k.split_interiors)
                continue
            actual_count = gc.cindex.get_value()[0]
            actual_count = np.sum(actual_count == OcgisConvention.Value.MULTI_BREAK_VALUE) + 1
            self.assertEqual(actual_count, desired_count)

    def test_convert_to_geometry_coordinates_polygons(self):
        grid = create_gridxy_global(resolution=45.0)
        geom = grid.get_abstraction_geometry()
        geom.reshape(Dimension('n_elements', geom.size))

        keywords = dict(pack=[True, False], repeat_last_node=[False, True], start_index=[0, 1])
        for k in self.iter_product_keywords(keywords):
            actual = geom.convert_to(pack=k.pack, repeat_last_node=k.repeat_last_node, start_index=k.start_index)
            self.assertEqual(actual.cindex.attrs['start_index'], k.start_index)
            self.assertEqual(actual.start_index, k.start_index)
            self.assertEqual(actual.cindex.get_value()[0].min(), k.start_index)
            self.assertEqual(actual.packed, k.pack)
            self.assertIsNotNone(actual.cindex)

            for actual_geom, desired_geom in zip(actual.get_geometry_iterable(), geom.get_value().flat):
                self.assertEqual(actual_geom[1], desired_geom)

    def test_convert_to_self_intersecting(self):
        poly = Polygon(self.fixture_self_intersecting_polygon_coords)
        gvar = GeometryVariable.from_shapely(poly)
        without_si = gvar.convert_to(remove_self_intersects=True)
        gvar2 = without_si.convert_to()
        desired = do_remove_self_intersects_multi(poly)
        self.assertPolygonSimilar(gvar2.v()[0], desired)
        # gvar2.write_vector('/tmp/without_si.shp')

    @attr('mpi')
    def test_create_ugid_global(self):
        ompi = OcgDist()
        m = ompi.create_dimension('m', 4)
        n = ompi.create_dimension('n', 70, dist=True)
        ompi.update_dimension_bounds()

        gvar = GeometryVariable(name='geom', dimensions=(m, n))
        ugid = gvar.create_ugid_global('gid')

        if not gvar.is_empty:
            self.assertEqual(gvar.dist, ugid.dist)

        gathered = variable_gather(ugid)
        if MPI_RANK == 0:
            actual = gathered.get_value()
            self.assertEqual(actual.size, len(set(actual.flatten().tolist())))

    def test_crs(self):
        crs = WGS84()
        gvar = GeometryVariable(crs=crs, name='var')
        self.assertEqual(gvar.crs, crs)
        self.assertIn(crs.name, gvar.parent)
        gvar.crs = None
        self.assertIsNone(gvar.crs)
        self.assertEqual(len(gvar.parent), 1)
        self.assertNotIn(crs.name, gvar.parent)

        # Test coordinate system is maintained.
        gvar = GeometryVariable(crs=crs, name='var')
        vc = VariableCollection(variables=gvar)
        self.assertIn(crs.name, vc)

    def test_deepcopy(self):
        gvar = GeometryVariable(value=Point(1, 2), crs=Spherical(), dimensions='d')
        self.assertEqual(gvar.crs, Spherical())
        d = gvar.deepcopy()
        d.crs = WGS84()
        self.assertEqual(gvar.crs, Spherical())
        self.assertFalse(np.may_share_memory(gvar.get_value(), d.get_value()))
        self.assertIsNotNone(d.crs)

    def test_getitem(self):
        gridxy = self.get_gridxy()
        pa = get_geometry_variable(gridxy)
        self.assertEqual(pa.shape, (4, 3))
        self.assertEqual(pa.ndim, 2)
        self.assertIsNotNone(pa._value)
        sub = pa[2:4, 1]
        self.assertEqual(sub.shape, (2, 1))
        self.assertEqual(sub.get_value().shape, (2, 1))

        # Test slicing with a parent.
        pa = self.get_geometryvariable_with_parent()
        desired_obj = pa.parent['tas']
        self.assertIsNotNone(pa.parent)
        desired = desired_obj[:, 1].get_value()
        self.assertIsNotNone(pa.parent)
        desired_shapes = OrderedDict([('tas', (10, 3)), ('point', (3,))])
        self.assertEqual(pa.parent.shapes, desired_shapes)

        sub = pa[1]
        backref_tas = sub.parent['tas']
        self.assertNumpyAll(backref_tas.get_value(), desired)
        self.assertEqual(backref_tas.shape, (10, 1))

    def test_geom_type(self):
        gvar = GeometryVariable(value=Point(1, 2), dimensions='ngeom')
        self.assertEqual(gvar.geom_type, 'Point')

        # Test with a multi-geometry.
        mp = np.array([None])
        mp[0] = MultiPoint([Point(1, 2), Point(3, 4)])
        pa = self.get_geometryvariable(value=mp)
        self.assertEqual(pa.geom_type, 'MultiPoint')

        # Test overloading.
        pa = self.get_geometryvariable(value=mp, geom_type='overload')
        self.assertEqual(pa.geom_type, 'overload')

    @attr('mpi')
    def test_geom_type_global(self):
        if MPI_SIZE != 3:
            raise SkipTest('MPI_SIZE != 3')

        geoms = [Point(1, 2), MultiPoint([Point(3, 4)]), Point(5, 6)]
        geom = GeometryVariable(name='geom', value=[geoms[MPI_RANK]], dimensions='geom')
        self.assertEqual(geom.geom_type, geoms[MPI_RANK].geom_type)
        self.assertEqual(geom.geom_type_global, geoms[1].geom_type)

    @attr('mpi')
    def test_get_intersects(self):
        dist = OcgDist()
        dist.create_dimension('x', 5, dist=False)
        dist.create_dimension('y', 5, dist=True)
        dist.update_dimension_bounds()

        if MPI_RANK == 0:
            x = Variable(value=[1, 2, 3, 4, 5], name='x', dimensions=['x'])
            y = Variable(value=[10, 20, 30, 40, 50], name='y', dimensions=['y'])
        else:
            x, y = [None] * 2

        x = variable_scatter(x, dist)
        y = variable_scatter(y, dist)

        if MPI_RANK < 2:
            self.assertTrue(y.dimensions[0].dist)

        grid = Grid(x=x, y=y)
        if not grid.is_empty:
            self.assertTrue(grid.dimensions[0].dist)
        pa = get_geometry_variable(grid)
        if MPI_RANK >= 2:
            self.assertTrue(pa.is_empty)
        polygon = box(2.5, 15, 4.5, 45)
        if not grid.is_empty:
            self.assertTrue(pa.dimensions[0].dist)

        # if MPI_RANK == 0:
        #     self.write_fiona_htmp(GeometryVariable(value=polygon), 'polygon')
        # self.write_fiona_htmp(grid.abstraction_geometry, 'grid-{}'.format(MPI_RANK))

        # Try an empty subset.
        live_ranks = vm.get_live_ranks_from_object(pa)
        vm.create_subcomm('test_get_intersects', live_ranks, is_current=True)
        if not vm.is_null:
            with self.assertRaises(EmptySubsetError):
                pa.get_intersects(Point(-8000, 9000))

            sub, slc = pa.get_intersects(polygon, return_slice=True)
        else:
            sub, slc = [None] * 2

        # self.write_fiona_htmp(sub, 'sub-{}'.format(MPI_RANK))

        if MPI_SIZE == 1:
            self.assertEqual(sub.shape, (3, 2))
        else:
            # This is the non-distributed dimension.
            if MPI_SIZE == 2:
                self.assertEqual(sub.shape[1], 2)
            # This is the distributed dimension.
            if MPI_RANK in live_ranks:
                if MPI_RANK < 5:
                    self.assertNotEqual(sub.shape[0], 3)
                else:
                    self.assertTrue(sub.is_empty)

        if not vm.is_null:
            desired_points_slc = pa.get_distributed_slice(slc).get_value()

            desired_points_manual = [Point(x, y) for x, y in
                                     itertools.product(grid.x.get_value().flat, grid.y.get_value().flat)]
            desired_points_manual = [pt for pt in desired_points_manual if pt.intersects(polygon)]
            for desired_points in [desired_points_manual, desired_points_slc.flat]:
                for pt in desired_points:
                    found = False
                    for pt_actual in sub.get_value().flat:
                        if pt_actual.almost_equals(pt):
                            found = True
                            break
                    self.assertTrue(found)

        # Test w/out an associated grid.
        if not vm.is_null:
            pa = self.get_geometryvariable(dimensions='ngeom')
            polygon = box(0.5, 1.5, 1.5, 2.5)
            sub = pa.get_intersects(polygon)
            self.assertEqual(sub.shape, (1,))
            self.assertEqual(sub.get_value()[0], Point(1, 2))

    def test_get_intersection(self):
        for return_indices in [True, False]:
            pa = self.get_geometryvariable(dimensions='ngeom')
            polygon = box(0.9, 1.9, 1.5, 2.5)
            lhs = pa.get_intersection(polygon, return_slice=return_indices)
            if return_indices:
                lhs, slc = lhs
                # self.assertEqual(slc, (slice(0, -1, None),))
            self.assertEqual(lhs.shape, (1,))
            self.assertEqual(lhs.get_value()[0], Point(1, 2))
            if return_indices:
                self.assertEqual(pa.get_value()[slc][0], Point(1, 2))

    @attr('mpi')
    def test_get_mask_from_intersects(self):
        poly = wkt.loads(
            'POLYGON((-98.26574367088608142 40.19952531645570559,-98.71764240506330168 39.54825949367089066,-99.26257911392406186 39.16281645569620906,-99.43536392405064817 38.64446202531645724,-98.78409810126584034 38.33876582278481493,-98.23916139240508016 37.71408227848101546,-97.77397151898735217 37.67420886075949937,-97.62776898734178133 38.15268987341772799,-98.39865506329114453 38.52484177215190186,-98.23916139240508016 39.33560126582278826,-97.73409810126582897 39.58813291139241386,-97.52143987341773368 40.27927215189873777,-97.52143987341773368 40.27927215189873777,-98.26574367088608142 40.19952531645570559))')
        desired_mask = np.array([[True, True, False, True],
                                 [True, False, True, True],
                                 [True, True, False, True]])

        dist = OcgDist()
        xdim = dist.create_dimension('x', 4, dist=True)
        ydim = dist.create_dimension('y', 3)
        dist.create_dimension('bounds', 2)
        dist.update_dimension_bounds()

        if MPI_RANK == 0:
            x = self.get_variable_x()
            y = self.get_variable_y()
            grid = Grid(x=x, y=y, abstraction='point', crs=WGS84())
            pa = get_geometry_variable(grid)
        else:
            pa = None
        pa = variable_scatter(pa, dist)

        vm.create_subcomm_by_emptyable('test_get_mask_from_intersects', pa, is_current=True)
        if vm.is_null:
            self.assertTrue(pa.is_empty)
            return

        usi = [False]
        if env.USE_SPATIAL_INDEX:
            usi.append(True)

        keywords = dict(use_spatial_index=usi)
        for k in self.iter_product_keywords(keywords):
            ret = pa.get_mask_from_intersects(poly, use_spatial_index=k.use_spatial_index)
            desired_mask_local = desired_mask[slice(*ydim.bounds_local), slice(*xdim.bounds_local)]
            if MPI_RANK > 1:
                self.assertIsNone(ret)
            else:
                self.assertNumpyAll(desired_mask_local, ret)

            # This does not test a parallel operation.
            if MPI_RANK == 0:
                # Test pre-masked values in geometry are okay for intersects operation.
                value = [Point(1, 1), Point(2, 2), Point(3, 3)]
                value = np.ma.array(value, mask=[False, True, False], dtype=object)
                pa2 = GeometryVariable(value=value, dimensions='ngeom')
                b = box(0, 0, 5, 5)
                res = pa2.get_mask_from_intersects(b, use_spatial_index=k.use_spatial_index)
                self.assertNumpyAll(res, value.mask)

    def test_get_nearest(self):
        target1 = Point(0.5, 0.75)
        target2 = box(0.5, 0.75, 0.55, 0.755)
        pa = self.get_geometryvariable()
        for target in [target1, target2]:
            res, slc = pa.get_nearest(target, return_indices=True)
            self.assertIsInstance(res, GeometryVariable)
            self.assertEqual(res.get_value()[0], Point(1, 2))
            self.assertEqual(slc, (0,))
            self.assertEqual(res.shape, (1,))

    @attr('rtree')
    def test_get_spatial_index(self):
        from ocgis.spatial.index import SpatialIndex

        pa = self.get_geometryvariable()
        si = pa.get_spatial_index()
        self.assertIsInstance(si, SpatialIndex)
        self.assertEqual(si._index.bounds, [1.0, 2.0, 3.0, 4.0])

    def test_get_split_polygon_by_node_threshold(self):
        geom = wkt.loads(strings.S7)
        desired_areas = [8.625085418953529e-08, 3.0976520763574336e-05, 6.726446875804517e-05, 5.750018860488541e-05,
                         2.5692352392841386e-05, 1.622423300407803e-12, 1.9296558045857587e-05, 5.476622145298075e-07,
                         1.5914858342000236e-05, 7.992538906530605e-05, 8.189708024845378e-05, 8.189708024845378e-05,
                         8.16714211647273e-05, 6.424884799935865e-05, 4.07608383401559e-06, 1.931694740983377e-05,
                         8.187851177678185e-05, 8.189708024845378e-05, 8.189708024845378e-05, 8.189708024832892e-05,
                         8.189708024845378e-05, 6.676074368443212e-05, 7.649059989813489e-06, 6.346561992475539e-05,
                         8.189708024832892e-05, 8.189708024845378e-05, 8.189708024845378e-05, 8.189708024832892e-05,
                         8.189708024845378e-05, 8.189708024845378e-05, 5.283621893398481e-05, 9.803354095522107e-06,
                         7.773749206872697e-06, 7.372953225573074e-05, 8.189708024852e-05, 8.189708024852e-05,
                         8.189708024839513e-05, 8.189708024852e-05, 8.189708024852e-05, 8.189708024852e-05,
                         4.113936769606091e-05, 1.3049151368967734e-05, 2.9750354449108212e-05, 7.809033449523e-05,
                         8.189708024832892e-05, 8.189708024845378e-05, 8.189708024845378e-05, 8.189708024845378e-05,
                         8.05761329781604e-05, 4.2576133026178655e-05, 2.4556126555499892e-05, 6.596040883756183e-05,
                         8.189708024845378e-05, 8.189708024845378e-05, 8.189708024845378e-05, 8.180315387085656e-05,
                         6.260717702153742e-05, 5.064617167749921e-05, 8.189708024845378e-05, 8.189708024845378e-05,
                         8.091143156873771e-05, 3.557751568323601e-05, 4.6024600921662305e-07, 3.389201434711213e-05,
                         8.189708024845378e-05, 7.696909562148969e-05, 6.081534208401692e-05, 5.3918215795366603e-08,
                         5.115495958841281e-06, 5.7158055052022955e-05, 6.446799906331681e-06]

        actual = get_split_polygon_by_node_threshold(geom, 10)

        # self.remove_dir = False
        # import os
        # to_write = GeometryVariable.from_shapely(actual, crs=Spherical())
        # to_write.write_vector(os.path.join(self.current_dir_output, 'split.shp'))
        # print(self.current_dir_output)

        self.assertAlmostEqual(geom.area, actual.area)

        actual_areas = [g.area for g in actual]
        for idx in range(len(desired_areas)):
            self.assertAlmostEqual(actual_areas[idx], desired_areas[idx])

    def test_get_unioned(self):
        # TODO: Test with an n-dimensional mask.

        ancillary = Variable('ancillary')
        pa = self.get_geometryvariable(parent=ancillary.parent, crs=WGS84())

        self.assertIn(ancillary.name, pa.parent)

        unioned = pa.get_unioned()
        new_uid = Variable('flower', value=[100], dimensions=unioned.dimensions)
        unioned.set_ugid(new_uid)

        # Parent should be removed from the unioned variable.
        self.assertNotIn(ancillary.name, unioned.parent)
        self.assertIn(ancillary.name, pa.parent)

        self.assertEqual(unioned.crs, WGS84())
        self.assertEqual(unioned.shape, (1,))
        desired = MultiPoint([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(unioned.get_value()[0], desired)
        self.assertEqual(len(unioned.dimensions[0]), 1)
        self.assertIsNone(unioned.get_mask())
        self.assertEqual(unioned.ugid.get_value()[0], 100)
        self.assertNotEqual(id(unioned), id(pa))

    def test_get_unioned_spatial_average(self):
        pa = self.get_geometryvariable()
        to_weight = Variable(name='to_weight', dimensions=pa.dimensions, dtype=float)
        to_weight.get_value()[:] = 5.0
        pa.parent.add_variable(to_weight)
        unioned = pa.get_unioned(spatial_average='to_weight')
        self.assertEqual(unioned.parent[to_weight.name].get_value().tolist(), [5.0])
        self.assertEqual(pa.parent[to_weight.name].get_value().shape, (2,))
        self.assertEqual(unioned.dimensions, unioned.parent[to_weight.name].dimensions)
        self.assertEqual(id(unioned.dimensions[0]), id(unioned.parent[to_weight.name].dimensions[0]))

    def test_get_unioned_spatial_average_differing_dimensions(self):
        pa = self.get_geometryvariable()

        to_weight = Variable(name='to_weight', dimensions=pa.dimensions, dtype=float)
        to_weight.get_value()[0] = 5.0
        to_weight.get_value()[1] = 10.0
        pa.parent.add_variable(to_weight)

        to_weight2 = Variable(name='to_weight2',
                              dimensions=[Dimension('time', 10), Dimension('level', 3), pa.dimensions[0]], dtype=float)
        for time_idx in range(to_weight2.shape[0]):
            for level_idx in range(to_weight2.shape[1]):
                to_weight2.get_value()[time_idx, level_idx] = (time_idx + 2) + (level_idx + 2) ** (level_idx + 1)
        pa.parent.add_variable(to_weight2)

        unioned = pa.get_unioned(spatial_average=['to_weight', 'to_weight2'])

        actual = unioned.parent[to_weight2.name]
        self.assertEqual(actual.shape, (10, 3, 1))
        self.assertEqual(to_weight2.shape, (10, 3, 2))
        self.assertNumpyAll(actual.get_value(), to_weight2.get_value()[:, :, 0].reshape(10, 3, 1))
        self.assertEqual(actual.dimension_names, ('time', 'level', 'ocgis_geom_union'))

        self.assertEqual(unioned.parent[to_weight.name].get_value()[0], 7.5)

    @attr('mpi')
    def test_get_unioned_spatial_average_parallel(self):
        if MPI_SIZE != 8:
            raise SkipTest('MPI_SIZE != 8')

        dist = OcgDist()
        geom_count = dist.create_dimension('geom_count', size=8, dist=True)
        time_count = dist.create_dimension('time', size=3)
        dist.update_dimension_bounds()

        if not geom_count.is_empty:
            gvar = GeometryVariable(value=[Point(1.0, 1.0).buffer(MPI_RANK + 2)] * len(geom_count),
                                    dimensions=geom_count)
            value = np.zeros((len(time_count), len(geom_count)), dtype=float)
            for ii in range(value.shape[0]):
                value[ii] = [MPI_RANK + 1 + ii + 1] * len(geom_count)
            data = Variable(name='data', value=value, dtype=float, dimensions=[time_count, geom_count])
        else:
            gvar = GeometryVariable(dimensions=geom_count)
            data = Variable(name='data', dimensions=[time_count, geom_count], dtype=float)
        gvar.parent.add_variable(data)

        self.assertTrue(gvar.is_empty == data.is_empty == gvar.parent.is_empty)

        with vm.scoped_by_emptyable('union', gvar):
            if vm.is_null:
                unioned = None
            else:
                unioned = gvar.get_unioned(spatial_average='data')

        if unioned is not None:
            self.assertIsInstance(unioned, GeometryVariable)
            actual = unioned.parent[data.name]
            self.assertAlmostEqual(actual.get_value().max(), 5.5466666666666677)
        else:
            self.assertIsNone(unioned)

    def test_prepare(self):
        coords = (-10.0, 40.0, 50.0, 50.0)
        bbox = box(*coords)
        gvar = GeometryVariable(value=bbox, dimensions='geom', crs=Spherical(), is_bbox=True,
                                wrapped_state=WrappedState.UNWRAPPED)

        for _ in range(3):
            prepared = gvar.prepare()
            self.assertNotEqual(id(prepared), id(gvar))
            actual = []
            desired = [(-10.0, 40.0, 50.0, 50.0), (350.0, 40.0, 370.0, 50.0)]
            for g in prepared.get_value().flatten()[0]:
                actual.append(g.bounds)
            self.assertEqual(actual, desired)

        # Test updating the coordinate system.
        updated_crs = [WGS84, Spherical, None]
        wrapped_states = [WrappedState.WRAPPED, WrappedState.UNWRAPPED, WrappedState.UNKNOWN]
        keywords = dict(wrapped_state_src=wrapped_states,
                        wrapped_state_dst=wrapped_states,
                        crs_src=updated_crs,
                        crs_dst=updated_crs)
        for ctr, k in enumerate(self.iter_product_keywords(keywords)):
            dgvar = deepcopy(gvar)
            dgvar.wrapped_state = k.wrapped_state_src
            if k.crs_src is None:
                dgvar.crs = None
            else:
                dgvar.crs = k.crs_src()

            archetype = deepcopy(gvar)
            archetype.wrapped_state = k.wrapped_state_dst
            if k.crs_dst is None:
                archetype.crs = None
            else:
                archetype.crs = k.crs_dst()

            actual = dgvar.prepare(archetype=archetype)

            if k.crs_src is not None and k.crs_dst is not None:
                self.assertEqual(actual.crs, archetype.crs)
                if dgvar.wrapped_state not in (WrappedState.UNKNOWN,) and archetype.wrapped_state not in (WrappedState.UNKNOWN,):
                    self.assertEqual(actual.wrapped_state, archetype.wrapped_state)

        # Test identical object is returned if nothing happens.
        dgvar = deepcopy(gvar)
        dgvar.crs = None
        actual = dgvar.prepare()
        self.assertNotEqual(id(actual), id(dgvar))
        self.assertEqual(id(dgvar.v()[0]), id(actual.v()[0]))

        # Test exception for more than one geometry.
        gvar = mock.create_autospec(GeometryVariable, spec_set=True)
        gvar.size = 2
        with self.assertRaises(RequestableFeature):
            GeometryVariable.prepare(gvar)

    def test_unwrap(self):
        geom = box(195, -40, 225, -30)
        gvar = GeometryVariable(name='geoms', value=geom, crs=Spherical(), dimensions='geoms')
        gvar.wrap()
        self.assertEqual(gvar.get_value()[0].bounds, (-165.0, -40.0, -135.0, -30.0))
        gvar.unwrap()
        self.assertEqual(gvar.get_value()[0].bounds, (195.0, -40.0, 225.0, -30.0))

    def test_update_crs(self):
        from_crs = WGS84()
        pa = self.get_geometryvariable(crs=from_crs, name='g', dimensions='gg')
        to_crs = CoordinateReferenceSystem(epsg=2136)
        pa.update_crs(to_crs)
        self.assertEqual(pa.crs, to_crs)
        v0 = [1629871.494956261, -967769.9070825744]
        v1 = [2358072.3857447207, -239270.87548993886]
        np.testing.assert_almost_equal(pa.get_value()[0], v0, decimal=3)
        np.testing.assert_almost_equal(pa.get_value()[1], v1, decimal=3)

    def test_update_crs_to_cartesian(self):
        """Test a spherical to cartesian CRS update."""

        bbox = box(-170., 40., 150., 80.)
        original_bounds = deepcopy(bbox.bounds)
        geom = GeometryVariable(name='geom', value=[bbox], dimensions='geom', crs=Spherical())

        other_crs = Cartesian()
        geom.update_crs(other_crs)
        actual = geom.get_value()[0].bounds
        desired = (-0.7544065067354889, -0.13302222155948895, -0.15038373318043535, 0.38302222155948895)
        self.assertNumpyAllClose(np.array(actual), np.array(desired))
        self.assertIsInstance(geom.crs, Cartesian)

        other_crs = Spherical()
        geom.update_crs(other_crs)
        self.assertEqual(geom.crs, Spherical())
        actual = geom.get_value()[0].bounds
        self.assertNumpyAllClose(np.array(original_bounds), np.array(actual))

        # Test data may not be wrapped.
        bbox = box(0, 40, 270, 80)
        geom = GeometryVariable(name='geom', value=[bbox], dimensions='geom', crs=Spherical())
        other_crs = Cartesian()
        with self.assertRaises(ValueError):
            geom.update_crs(other_crs)

    def test_weights(self):
        value = [Point(2, 3), Point(4, 5), Point(5, 6)]
        mask = [False, True, False]
        value = np.ma.array(value, mask=mask, dtype=object)
        pa = self.get_geometryvariable(value=value)
        self.assertNumpyAll(pa.weights, np.ma.array([1, 1, 1], mask=mask, dtype=env.NP_FLOAT))

    def test_wrap(self):
        geom = box(195, -40, 225, -30)
        gvar = GeometryVariable(name='geoms', value=geom, crs=Spherical(), dimensions='geoms')
        gvar.wrap()
        self.assertEqual(gvar.get_value()[0].bounds, (-165.0, -40.0, -135.0, -30.0))


class TestGeometryVariablePolygons(AbstractTestInterface):
    """Test a geometry variable using polygons."""

    def test_init(self):
        row = Variable(value=[2, 3], name='row', dimensions='y')
        col = Variable(value=[4, 5], name='col', dimensions='x')
        grid = Grid(col, row)
        self.assertIsNone(grid.archetype.bounds)

        row = Variable(value=[2, 3], name='row', dimensions='y')
        row.set_extrapolated_bounds('row_bounds', 'bounds')
        col = Variable(value=[4, 5], name='col', dimensions='x')
        col.set_extrapolated_bounds('col_bounds', 'bounds')
        grid = Grid(y=row, x=col)
        self.assertEqual(grid.abstraction, 'polygon')
        poly = get_geometry_variable(grid)
        self.assertEqual(poly.geom_type, 'Polygon')

    def test_area_and_weights(self):
        poly = self.get_polygonarray()
        bbox = box(-98.1, 38.3, -99.4, 39.9)

        sub = poly.get_intersection(bbox)
        actual_area = [[0.360000000000001, 0.1600000000000017], [0.9000000000000057, 0.4000000000000057],
                       [0.18000000000000368, 0.08000000000000228]]
        np.testing.assert_almost_equal(sub.area, actual_area)
        actual_weights = [[0.3999999999999986, 0.17777777777777853], [1.0, 0.444444444444448],
                          [0.20000000000000284, 0.08888888888889086]]
        np.testing.assert_almost_equal(sub.weights, actual_weights)
