import itertools
from collections import OrderedDict
from copy import deepcopy

import numpy as np
from nose.plugins.skip import SkipTest
from numpy.ma import MaskedArray
from shapely import wkt
from shapely.geometry import Point, box, MultiPoint, LineString
from shapely.geometry.multilinestring import MultiLineString

from ocgis import RequestDataset, vm
from ocgis import env, CoordinateReferenceSystem
from ocgis.constants import DMK
from ocgis.exc import EmptySubsetError
from ocgis.spatial.grid import Grid, get_geometry_variable
from ocgis.test.base import attr, AbstractTestInterface
from ocgis.variable.base import Variable, VariableCollection
from ocgis.variable.crs import WGS84, Spherical, Cartesian
from ocgis.variable.dimension import Dimension
from ocgis.variable.geom import GeometryVariable, GeometryProcessor
from ocgis.vmachine.mpi import OcgDist, MPI_RANK, variable_scatter, MPI_SIZE, variable_gather, MPI_COMM


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


class TestGeometryVariable(AbstractTestInterface):
    @staticmethod
    def get_geometryvariable_with_parent():
        vpa = np.array([None, None, None])
        vpa[:] = [Point(1, 2), Point(3, 4), Point(5, 6)]
        value = np.arange(0, 30).reshape(10, 3)
        tas = Variable(name='tas', value=value, dimensions=['time', 'ngeom'])
        backref = VariableCollection(variables=[tas])
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

    def test_area(self):
        gvar = self.get_geometryvariable()
        self.assertTrue(np.all(gvar.area == 0))

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

    def test_get_unioned(self):
        # tdk: test with ndimensional mask

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
        row.set_extrapolated_bounds('row_bounds', 'y')
        col = Variable(value=[4, 5], name='col', dimensions='x')
        col.set_extrapolated_bounds('col_bounds', 'x')
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
