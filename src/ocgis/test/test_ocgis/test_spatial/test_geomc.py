from unittest.case import SkipTest

import numpy as np
from mock import mock
from ocgis import Variable, Dimension, vm, Field, GeometryVariable, DimensionMap
from ocgis.base import AbstractOcgisObject, raise_if_empty
from ocgis.constants import WrappedState, DMK, GridAbstraction, Topology, DriverKey, AttributeName
from ocgis.driver.nc_ugrid import DriverNetcdfUGRID
from ocgis.spatial.base import create_spatial_mask_variable
from ocgis.spatial.geomc import PointGC, get_default_geometry_variable_name, PolygonGC, reduce_reindex_coordinate_index, \
    iter_multipart_coordinates
from ocgis.spatial.grid_chunker import GridChunker
from ocgis.test.base import TestBase, attr
from ocgis.test.test_ocgis.test_driver.test_nc_ugrid import get_ugrid_data_structure
from ocgis.variable.crs import Spherical, WGS84
from ocgis.vmachine.mpi import OcgDist, variable_collection_scatter, variable_gather, variable_scatter, hgather
from shapely import wkt
from shapely.geometry import Point, MultiPolygon, box
from shapely.geometry.polygon import Polygon


class Test(TestBase):
    def test_iter_multipart_coordinates(self):
        arr = np.array([1, 2, -10, 3, 4, -10, 5, 6])
        for ctr, part in enumerate(iter_multipart_coordinates(arr, -10)):
            self.assertEqual(part.shape[0], 2)
        self.assertEqual(ctr, 2)

        arr = np.array([1, 2])
        actual = list(iter_multipart_coordinates(arr, -100))
        self.assertEqual(len(actual), 1)
        self.assertNumpyAll(actual[0], arr)

    @attr('mpi')
    def test_reduce_reindex_coordinate_index(self):
        dist = OcgDist()
        dist.create_dimension('dim', 12, dist=True)
        dist.update_dimension_bounds()

        global_cindex_arr = np.array([4, 2, 1, 2, 1, 4, 1, 4, 2, 5, 6, 7])

        if vm.rank == 0:
            var_cindex = Variable('cindex', value=global_cindex_arr, dimensions='dim')
        else:
            var_cindex = None
        var_cindex = variable_scatter(var_cindex, dist)

        vm.create_subcomm_by_emptyable('test', var_cindex, is_current=True)
        if vm.is_null:
            return

        raise_if_empty(var_cindex)

        coords = np.array([0, 11, 22, 33, 44, 55, 66, 77, 88, 99, 100, 110, 120, 130, 140, 150])
        coords = Variable(name='coords', value=coords, dimensions='coord_dim')

        new_cindex, u_indices = reduce_reindex_coordinate_index(var_cindex)

        desired = coords[global_cindex_arr].get_value()

        if len(u_indices) > 0:
            new_coords = coords[u_indices].get_value()
        else:
            new_coords = np.array([])
        gathered_new_coords = vm.gather(new_coords)
        gathered_new_cindex = vm.gather(new_cindex)
        if vm.rank == 0:
            gathered_new_coords = hgather(gathered_new_coords)
            gathered_new_cindex = hgather(gathered_new_cindex)

            actual = gathered_new_coords[gathered_new_cindex]

            self.assertAsSetEqual(gathered_new_cindex.tolist(), [2, 1, 0, 3, 4, 5])
            desired_new_coords = [11, 22, 44, 55, 66, 77]
            self.assertAsSetEqual(gathered_new_coords.tolist(), desired_new_coords)
            self.assertEqual(len(gathered_new_coords), len(desired_new_coords))

            self.assertNumpyAll(actual, desired)

    @attr('slow', 'mpi')
    def test_reduce_reindex_coordinate_index_stress(self):
        if vm.size != 8:
            raise SkipTest('vm.size != 8')

        for _ in range(3):
            value = np.random.random_integers(1000, 10000, 1000)
            _ = reduce_reindex_coordinate_index(value)


class FixturePointGC(AbstractOcgisObject):
    def fixture(self, **kwargs):
        kwargs = kwargs.copy()
        node_dimension_name = kwargs.pop('node_dimension_name', 'dnodes')

        x = Variable(name='xc', value=[0, 1, 2, 3, 4, 5], dimensions=node_dimension_name, dtype=float)
        y = Variable(name='yc', value=[6, 7, 8, 9, 10, 11], dimensions=node_dimension_name, dtype=float)
        z = Variable(name='zc', value=[12, 13, 14, 15, 16, 17], dimensions=node_dimension_name, dtype=float)

        p = PointGC(x, y, z=z, **kwargs)
        return p

    @staticmethod
    def fixture_cindex(start_index):
        assert start_index == 0 or start_index == 1
        value = np.arange(0 + start_index, 6 + start_index).reshape(-1, 1)
        return Variable(name='cindex',
                        value=value,
                        dimensions=['elements', 'misc'],
                        attrs={AttributeName.START_INDEX: start_index})

    @property
    def fixture_element_dimension(self):
        return Dimension('elements', size=6)

    @property
    def fixture_mask(self):
        return create_spatial_mask_variable('mask', [False, False, False, False, True, False],
                                            dimensions=self.fixture_element_dimension)


class TestPointGC(TestBase, FixturePointGC):
    @property
    def fixture_intersects_polygon(self):
        c = [[3.1, 7.9, 13.9],
             [3.1, 9.1, 13.9],
             [1.9, 9.1, 13.9],
             [1.9, 7.9, 13.9],

             [1.9, 7.9, 14.1],
             [3.1, 7.9, 14.1],
             [3.1, 9.1, 14.1],
             [1.9, 9.1, 14.1]]
        poly = Polygon(c)
        return poly

    def test_init(self):
        element_dim = self.fixture_element_dimension
        cindex = self.fixture_cindex(0)
        mask = self.fixture_mask
        keywords = dict(cindex=[None, cindex],
                        mask=[None, mask])

        for k in self.iter_product_keywords(keywords):
            p = self.fixture(cindex=k.cindex, mask=k.mask)
            self.assertEqual(p.parent.driver.key, DriverKey.NETCDF_UGRID)

            if k.cindex is None:
                self.assertIsNone(p.dimension_map.get_variable(DMK.X))
                dimension_map = p.dimension_map.get_topology(GridAbstraction.POINT)
                self.assertIsNotNone(dimension_map.get_variable(DMK.X))
                self.assertEqual(p.element_dim, p.archetype.dimensions[0])
            else:
                self.assertEqual(p.element_dim, element_dim)

            if k.cindex is None:
                actual = p.cindex
                desired = None
            else:
                self.assertEqual(p.cindex.attrs['start_index'], 0)
                actual = p.cindex.name
                desired = cindex.name
            self.assertEqual(actual, desired)

            if k.mask is None:
                self.assertFalse(p.has_mask)
            else:
                self.assertTrue(p.has_mask)

    def test_init_dimension_map(self):
        """Test initializing with a dimension map only."""

        dmap = DimensionMap()
        x = Variable(value=[1, 2, 3], dimensions='elements', name='x')
        y = Variable(value=[4, 5, 6], dimensions='elements', name='y')
        topo = dmap.get_topology(Topology.POINT, create=True)
        topo.set_variable(DMK.X, x)
        topo.set_variable(DMK.Y, y)
        f = Field(variables=[x, y], dimension_map=dmap)
        p = PointGC(parent=f)
        self.assertNumpyAll(x.get_value(), p.x.get_value())
        self.assertNumpyAll(y.get_value(), p.y.get_value())

    def test_init_driver(self):
        driver = DriverNetcdfUGRID
        pgc = self.fixture(driver=driver)
        self.assertEqual(pgc.parent.driver, driver)

    def test_convert_to(self):
        desired = WGS84()
        gc = self.fixture(crs=desired)
        gv = gc.convert_to()
        self.assertEqual(gv.crs, desired)

    def test_get_distributed_slice(self):
        f = self.fixture()
        slc = np.zeros(f.archetype.shape[0], dtype=bool)
        actual = f.get_distributed_slice(slc)
        self.assertTrue(actual.is_empty)

    def test_get_intersects(self):
        p = self.fixture(cindex=self.fixture_cindex(0))
        poly = self.fixture_intersects_polygon

        sub = p.get_intersects(poly)

        desired = Point(2, 8, 14)
        actual = list(sub.get_geometry_iterable())
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0][1], desired)

    def test_get_intersects_wrapped(self):
        """Test using a subset geometry that needs to be wrapped."""

        coords = (-10.0, 40.0, 50.0, 50.0)
        bbox = box(*coords)
        gvar = GeometryVariable(value=bbox, dimensions='geom', crs=Spherical(), is_bbox=True,
                                wrapped_state=WrappedState.UNWRAPPED)
        x = [60.0, 355.0]
        x = Variable(name='x', value=x, dimensions='n_nodes')

        y = [45.0, 45.0]
        y = Variable(name='y', value=y, dimensions='n_nodes')

        pgc = PointGC(x, y)
        ret = pgc.get_intersects(gvar)

        self.assertEqual(ret.archetype.size, 1)
        geom = list(ret.iter_geometries())[0][1]
        actual = np.array(geom).tolist()
        self.assertEqual(actual, [355., 45.])

    def test_get_intersection(self):
        p = self.fixture()
        poly = self.fixture_intersects_polygon
        desired = Point(2, 8, 14)

        keywords = {'geom_name': [None, 'output_geom'],
                    'optimized_bbox_subset': [False, True]}
        for k in self.iter_product_keywords(keywords):
            try:
                sub = p.get_intersection(poly, geom_name=k.geom_name, optimized_bbox_subset=k.optimized_bbox_subset)
            except ValueError:
                self.assertTrue(k.optimized_bbox_subset)
                continue

            if k.geom_name is None:
                desired_geom_name = get_default_geometry_variable_name(p)
            else:
                desired_geom_name = k.geom_name
            self.assertEqual(sub.name, desired_geom_name)
            actual = sub.get_value()[0]
            self.assertEqual(sub.size, 1)
            self.assertEqual(actual, desired)

    def test_gc_nchunks_dst(self):
        pgc = self.fixture()
        gc = mock.create_autospec(GridChunker)
        actual = pgc._gc_nchunks_dst_(gc)
        self.assertIsNotNone(actual)
        self.assertEqual(actual, (100,))

    def test_iter_geometries(self):
        keywords = dict(umo=[None, False, True],
                        cindex=[None, self.fixture_cindex(0)])
        for k in self.iter_product_keywords(keywords):
            p = self.fixture(mask=self.fixture_mask, cindex=k.cindex)
            actual = list(p.iter_geometries(use_memory_optimizations=k.umo))
            self.assertEqual(len(actual), p.archetype.shape[0])

            actual2 = np.array(actual[2][1]).tolist()
            desired = [2.0, 8.0, 14.0]
            self.assertEqual(actual2, desired)

            actual3 = actual[4][1]
            self.assertIsNone(actual3)

    def test_iter_geometries_with_cindex(self):
        cindex = Variable(name='cindex', value=[2, 2], dimensions='elements')
        p = self.fixture(cindex=cindex)
        self.assertIsNotNone(p.cindex)
        actual = list(p.iter_geometries())
        actual = np.array([ii[1].xy for ii in actual]).tolist()
        desired = [[[2.0], [8.0]], [[2.0], [8.0]]]
        self.assertEqual(actual, desired)

    @attr('mpi')
    def test_reduce_global(self):
        pt = self.fixture(cindex=self.fixture_cindex(1), start_index=1)
        self.assertEqual(pt.start_index, 1)

        dist = OcgDist()
        for d in pt.parent.dimensions.values():
            d = d.copy()
            if d.name == self.fixture_element_dimension.name:
                d.dist = True
            dist.add_dimension(d)
        dist.update_dimension_bounds()

        new_parent = variable_collection_scatter(pt.parent, dist)

        vm.create_subcomm_by_emptyable('coordinate reduction', new_parent, is_current=True)
        if vm.is_null:
            return

        pt.parent = new_parent
        sub = pt.get_distributed_slice(slice(2, 5))

        vm.create_subcomm_by_emptyable('distributed slice', sub, is_current=True)
        if vm.is_null:
            return

        actual = sub.reduce_global()

        actual_cindex = actual.cindex.extract()
        actual_cindex = variable_gather(actual_cindex)
        if vm.rank == 0:
            actual_cindex = actual_cindex.get_value().flatten().tolist()
            self.assertEqual(actual_cindex, [1, 2, 3])

        gathered = [variable_gather(c.extract()) for c in actual.coordinate_variables]
        if vm.rank == 0:
            actual_coords = []
            for c in gathered:
                actual_coords.append(c.get_value().tolist())
            desired = [[2.0, 3.0, 4.0], [8.0, 9.0, 10.0], [14.0, 15.0, 16.0]]
            self.assertEqual(actual_coords, desired)

        path = self.get_temporary_file_path('foo.nc')
        actual.parent.write(path)

        actual = Field.read(path)
        self.assertEqual(actual['cindex'].attrs['start_index'], 1)

        # if vm.rank == 0: self.ncdump(path, header_only=False)

    def test_update_crs(self):
        f = self.fixture(crs=Spherical())
        to_crs = WGS84()
        orig_diff = np.sum(np.diff(f.y.get_value()))
        f.update_crs(to_crs)
        actual_diff = np.sum(np.diff(f.y.get_value()))
        self.assertGreater(np.abs(orig_diff - actual_diff), 0.01)


class FixturePolygonGC(AbstractOcgisObject):
    def fixture(self):
        u = get_ugrid_data_structure()
        u = Field.from_variable_collection(u)
        x = u['face_node_x']
        y = u['face_node_y']
        cindex = u['face_node_index']
        poly = PolygonGC(x, y, cindex=cindex, parent=u)
        return poly

    def fixture_subset_geom(self):
        e1 = 'Polygon ((9.48049363057324967 -40.86146496815286611, 8.19864649681528945 -42.86783439490445602, 12.32285031847133894 -43.03503184713375873, 12.32285031847133894 -43.03503184713375873, 9.48049363057324967 -40.86146496815286611))'
        e2 = 'Polygon ((19.12221337579618208 -48.55254777070064165, 18.09116242038216882 -50.05732484076433053, 20.34832802547770925 -50.00159235668790103, 19.12221337579618208 -48.55254777070064165))'
        subset_geom = MultiPolygon([wkt.loads(e) for e in [e1, e2]])
        return subset_geom


class TestPolygonGC(FixturePolygonGC, TestBase):
    def test_init(self):
        actual = self.fixture()
        actual.parent.dimension_map.set_driver(DriverNetcdfUGRID)
        self.assertEqual(actual.cindex.name, actual.parent.grid.cindex.name)

    def test_get_intersection(self):
        subset_geom = self.fixture_subset_geom()
        poly = self.fixture()

        isub = poly.get_intersection(subset_geom)

        self.assertIsInstance(isub, GeometryVariable)

        desired = [1.6695340074648009, 2.9828593804233616, 1.2616396357639024]
        for a, d in zip(isub.area.tolist(), desired):
            self.assertAlmostEqual(a, d)

    @attr('mpi')
    def test_get_intersects(self):
        subset_geom = self.fixture_subset_geom()
        poly = self.fixture()

        # Scatter the polygon geometry coordinates for the parallel case ===============================================

        dist = OcgDist()
        for d in poly.parent.dimensions.values():
            d = d.copy()
            if d.name == poly.dimensions[0].name:
                d.dist = True
            dist.add_dimension(d)
        dist.update_dimension_bounds()

        poly.parent = variable_collection_scatter(poly.parent, dist)

        vm.create_subcomm_by_emptyable('scatter', poly, is_current=True)
        if vm.is_null:
            return

        poly.parent._validate_()

        for v in poly.parent.values():
            self.assertEqual(id(v.parent), id(poly.parent))
            self.assertEqual(len(v.parent), len(poly.parent))

        # ==============================================================================================================

        # p = os.path.join('/tmp/subset_geom.shp')
        # s = GeometryVariable.from_shapely(subset_geom)
        # s.write_vector(p)
        # p = os.path.join('/tmp/poly.shp')
        # s = poly.convert_to()
        # s.write_vector(p)

        sub = poly.get_intersects(subset_geom)
        vm.create_subcomm_by_emptyable('after intersects', sub, is_current=True)
        if vm.is_null:
            return

        actual = []
        for g in sub.iter_geometries():
            if g[1] is not None:
                actual.append([g[1].centroid.x, g[1].centroid.y])
        desired = [[20.0, -49.5], [10.0, -44.5], [10.0, -39.5]]
        actual = vm.gather(actual)
        if vm.rank == 0:
            gactual = []
            for a in actual:
                for ia in a:
                    gactual.append(ia)
            self.assertEqual(gactual, desired)

        self.assertEqual(len(sub.parent), len(poly.parent))

        sub.parent._validate_()
        sub2 = sub.reduce_global()
        sub2.parent._validate_()

        # p = os.path.join('/tmp/sub.shp')
        # s = sub.convert_to()
        # s.write_vector(p)
        # p = os.path.join('/tmp/sub2.shp')
        # s = sub2.convert_to()
        # s.write_vector(p)

        # Gather then broadcast coordinates so all coordinates are available on each process.
        to_add = []
        for gather_target in [sub2.x, sub2.y]:
            gathered = variable_gather(gather_target.extract())
            gathered = vm.bcast(gathered)
            to_add.append(gathered)
        for t in to_add:
            sub2.parent.add_variable(t, force=True)

        for ctr, to_check in enumerate([sub, sub2]):
            actual = []
            for g in to_check.iter_geometries():
                if g[1] is not None:
                    actual.append([g[1].centroid.x, g[1].centroid.y])
            desired = [[20.0, -49.5], [10.0, -44.5], [10.0, -39.5]]
            actual = vm.gather(actual)
            if vm.rank == 0:
                gactual = []
                for a in actual:
                    for ia in a:
                        gactual.append(ia)
                self.assertEqual(gactual, desired)

                # # ============================================================================================================
                # import matplotlib.pyplot as plt
                # from descartes import PolygonPatch
                #
                # BLUE = '#6699cc'
                # GRAY = '#999999'
                #
                # fig = plt.figure(num=1)
                # ax = fig.add_subplot(111)
                #
                # polys = [g[1] for g in poly.iter_geometries()]
                # for p in polys:
                #     patch = PolygonPatch(p, fc=BLUE, ec=GRAY, alpha=0.5, zorder=1)
                #     ax.add_patch(patch)
                #
                # minx, miny, maxx, maxy = MultiPolygon(polys).bounds
                # w, h = maxx - minx, maxy - miny
                # ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
                # ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
                # ax.set_aspect(1)
                #
                # plt.show()

    def test_iter_geometries(self):
        poly = self.fixture()

        for g in poly.iter_geometries():
            self.assertIsInstance(g[0], int)
            self.assertIsInstance(g[1], Polygon)
