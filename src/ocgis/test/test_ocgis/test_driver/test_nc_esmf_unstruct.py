import itertools
import os
from copy import deepcopy
from unittest import SkipTest

import numpy as np
import shapely
from shapely import wkt
from shapely.geometry import MultiPolygon

import ocgis
from ocgis import GeometryVariable, vm, Field, Dimension
from ocgis.base import get_variable_names
from ocgis.constants import DriverKey, VariableName, Topology, DMK, AttributeName
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.geomc import PolygonGC
from ocgis.spatial.grid import GridUnstruct
from ocgis.test.base import TestBase, attr
from ocgis.variable.crs import WGS84, Spherical, create_crs
from ocgis.vmachine.mpi import OcgDist


class TestDriverESMFUnstruct(TestBase):

    @property
    def metadata_esmf_unstruct(self):
        d = {'groups': {},
             'global_attributes': {'gridType': 'unstructured mesh',
                                   'version': '0.9',
                                   'inputFile': 'grids/ll_grids/ll1280x1280_grid.nc',
                                   'timeGenerated': 'Tu'},
             'variables': {'nodeCoords': {'dimensions': ('nodeCount', 'coordDim'),
                                          'attrs': {'units': 'degrees'},
                                          'dtype': 'float64',
                                          'name': 'nodeCoords',
                                          'fill_value': 'auto',
                                          'dtype_packed': None,
                                          'fill_value_packed': None},
                           'elementConn': {'dimensions': ('elementCount', 'maxNodePElement'),
                                           'attrs': {'long_name': 'Node indices that define the element connectivity',
                                                     '_FillValue': -1},
                                           'dtype': 'int32',
                                           'name': 'elementConn',
                                           'fill_value': -1,
                                           'dtype_packed': None,
                                           'fill_value_packed': None},
                           'numElementConn': {'dimensions': ('elementCount',),
                                              'attrs': {'long_name': 'Number of nodes per element'},
                                              'dtype': 'int8',
                                              'name': 'numElementConn',
                                              'fill_value': 'auto',
                                              'dtype_packed': None,
                                              'fill_value_packed': None},
                           'centerCoords': {'dimensions': ('elementCount', 'coordDim'), 'attrs': {'units': 'degrees'},
                                            'dtype': 'float64', 'name': 'centerCoords', 'fill_value': 'auto',
                                            'dtype_packed': None, 'fill_value_packed': None},
                           'elementMask': {'dimensions': ('elementCount',), 'attrs': {}, 'dtype': 'int32',
                                           'name': 'elementMask', 'fill_value': 'auto', 'dtype_packed': None,
                                           'fill_value_packed': None}},
             'dimensions': {'nodeCount': {'name': 'nodeCount', 'size': 1639680, 'isunlimited': False},
                            'elementCount': {'name': 'elementCount', 'size': 1638400, 'isunlimited': False},
                            'maxNodePElement': {'name': 'maxNodePElement', 'size': 4, 'isunlimited': False},
                            'coordDim': {'name': 'coordDim', 'size': 2, 'isunlimited': False}}}
        return d

    @property
    def path_esmf_unstruct(self):
        return os.path.join(self.path_bin, 'nc', 'll1280x1280_grid.esmf.subset.nc')

    def fixture_esmf_unstruct_field(self):
        rd = RequestDataset(metadata=self.metadata_esmf_unstruct, driver=DriverESMFUnstruct)
        return rd.create_field()

    @attr('mpi', 'slow')
    def test_system_converting_state_boundaries_shapefile(self):
        verbose = False
        if verbose: ocgis.vm.barrier_print("starting test")
        ocgis.env.USE_NETCDF4_MPI = False  # tdk:FIX: this hangs in the STATE_FIPS write for asynch might be nc4 bug...
        keywords = {'transform_to_crs': [None, Spherical],
                    'use_geometry_iterator': [False, True]}
        actual_xsums = []
        actual_ysums = []
        for k in self.iter_product_keywords(keywords):
            if k.use_geometry_iterator and k.transform_to_crs is not None:
                to_crs = k.transform_to_crs()
            else:
                to_crs = None
            if k.transform_to_crs is None:
                desired_crs = WGS84()
            else:
                desired_crs = k.transform_to_crs()

            rd = RequestDataset(uri=self.path_state_boundaries, variable=['UGID', 'ID'])
            rd.metadata['schema']['geometry'] = 'MultiPolygon'
            field = rd.get()
            self.assertEqual(len(field.data_variables), 2)

            # Test there is no mask present.
            if verbose: ocgis.vm.barrier_print("before geom.load()")
            field.geom.load()
            if verbose: ocgis.vm.barrier_print("after geom.load()")
            self.assertFalse(field.geom.has_mask)
            self.assertNotIn(VariableName.SPATIAL_MASK, field)
            self.assertIsNone(field.dimension_map.get_spatial_mask())

            self.assertEqual(field.crs, WGS84())
            if k.transform_to_crs is not None:
                field.update_crs(desired_crs)
            self.assertEqual(len(field.data_variables), 2)
            self.assertEqual(len(field.geom.parent.data_variables), 2)
            if verbose: ocgis.vm.barrier_print("starting conversion")
            try:
                gc = field.geom.convert_to(pack=False, use_geometry_iterator=k.use_geometry_iterator, to_crs=to_crs)
            except ValueError as e:
                try:
                    self.assertFalse(k.use_geometry_iterator)
                    self.assertIsNotNone(to_crs)
                except AssertionError:
                    raise e
                else:
                    continue
            if verbose: ocgis.vm.barrier_print("after conversion")

            actual_xsums.append(gc.x.get_value().sum())
            actual_ysums.append(gc.y.get_value().sum())
            self.assertEqual(gc.crs, desired_crs)

            # Test there is no mask present after conversion to geometry coordinates.
            self.assertFalse(gc.has_mask)
            self.assertNotIn(VariableName.SPATIAL_MASK, gc.parent)
            self.assertIsNone(gc.dimension_map.get_spatial_mask())

            path = self.get_temporary_file_path('esmf_state_boundaries.nc')
            self.assertEqual(gc.parent.crs, desired_crs)
            gc.parent.write(path, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
            if verbose: ocgis.vm.barrier_print("after gc.parent.write")

            gathered_geoms = vm.gather(field.geom.get_value())
            if verbose: ocgis.vm.barrier_print("after gathered_geoms")
            with vm.scoped("gather test", [0]):
                if not vm.is_null:
                    actual_geoms = []
                    for g in gathered_geoms:
                        actual_geoms.extend(g)

                    rd = RequestDataset(path, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)
                    infield = rd.get()
                    self.assertEqual(create_crs(infield.crs.value), desired_crs)
                    for dv in field.data_variables:
                        self.assertIn(dv.name, infield)
                    ingrid = infield.grid
                    self.assertIsInstance(ingrid, GridUnstruct)

                    for g in ingrid.archetype.iter_geometries():
                        self.assertPolygonSimilar(g[1], actual_geoms[g[0]], check_type=False)
            if verbose: ocgis.vm.barrier_print("after gathered_geoms testing")

        vm.barrier()

        # Test coordinates have actually changed.
        if verbose: ocgis.vm.barrier_print("before use_geometry_iterator test")
        if not k.use_geometry_iterator:
            for ctr, to_test in enumerate([actual_xsums, actual_ysums]):
                for lhs, rhs in itertools.combinations(to_test, 2):
                    if ctr == 0:
                        self.assertAlmostEqual(lhs, rhs)
                    else:
                        self.assertNotAlmostEqual(lhs, rhs)

    def test_system_converting_state_boundaries_shapefile_memory(self):
        """Test iteration may be used in place of loading all values from source."""

        rd = RequestDataset(uri=self.path_state_boundaries)
        field = rd.get()
        data_variable_names = get_variable_names(field.data_variables)
        field.geom.protected = True
        sub = field.get_field_slice({'geom': slice(10, 20)})
        self.assertTrue(sub.geom.protected)
        self.assertFalse(sub.geom.has_allocated_value)

        self.assertIsInstance(sub, Field)
        self.assertIsInstance(sub.geom, GeometryVariable)
        gc = sub.geom.convert_to(use_geometry_iterator=True)
        self.assertIsInstance(gc, PolygonGC)

        self.assertFalse(sub.geom.has_allocated_value)
        self.assertTrue(field.geom.protected)
        path = self.get_temporary_file_path('out.nc')
        gc.parent.write(path)

    @attr('mpi')
    def test_system_spatial_subsetting(self):
        """Test spatial subsetting ESMF Unstructured format."""

        bbox = shapely.geometry.box(*[-119.2, 61.7, -113.2, 62.7])
        gvar = GeometryVariable(name='geom', value=bbox, is_bbox=True, dimensions='ngeom', crs=Spherical())
        gvar.unwrap()
        rd = RequestDataset(uri=self.path_esmf_unstruct,
                            driver=DriverESMFUnstruct,
                            crs=Spherical(),
                            grid_abstraction='point',
                            grid_is_isomorphic=True)
        field = rd.create_field()
        sub, slc = field.grid.get_intersects(gvar, optimized_bbox_subset=True, return_slice=True)
        desired_extent = np.array((240.890625, 61.8046875, 246.796875, 62.6484375))
        self.assertGreaterEqual(len(vm.get_live_ranks_from_object(sub)), 1)
        with vm.scoped_by_emptyable('reduction', sub):
            if not vm.is_null:
                red = sub.reduce_global()
                self.assertNumpyAllClose(desired_extent, np.array(red.extent_global))
        path = self.get_temporary_file_path('foo.nc', collective=True)
        with vm.scoped_by_emptyable('write', sub):
            if not vm.is_null:
                red.parent.write(path)

    @attr('mpi', 'esmf')
    def test_system_grid_chunking(self):
        if vm.size != 4: raise SkipTest('vm.size != 4')

        from ocgis.spatial.grid_chunker import GridChunker
        path = self.path_esmf_unstruct
        rd_dst = RequestDataset(uri=path,
                                driver=DriverESMFUnstruct,
                                crs=Spherical(),
                                grid_abstraction='point',
                                grid_is_isomorphic=True)
        rd_src = deepcopy(rd_dst)
        resolution = 0.28125
        chunk_wd = os.path.join(self.current_dir_output, 'chunks')
        if vm.rank == 0:
            os.mkdir(chunk_wd)
        vm.barrier()
        paths = {'wd': chunk_wd}
        gc = GridChunker(rd_src, rd_dst, nchunks_dst=[8], src_grid_resolution=resolution,
                         dst_grid_resolution=resolution,
                         optimized_bbox_subset=True, paths=paths, genweights=True)
        gc.write_chunks()

        dist = OcgDist()
        local_ctr = Dimension(name='ctr', size=8, dist=True)
        dist.add_dimension(local_ctr)
        dist.update_dimension_bounds()
        for ctr in range(local_ctr.bounds_local[0], local_ctr.bounds_local[1]):
            ctr += 1
            s = os.path.join(chunk_wd, 'split_src_{}.nc'.format(ctr))
            d = os.path.join(chunk_wd, 'split_dst_{}.nc'.format(ctr))
            sf = Field.read(s, driver=DriverESMFUnstruct)
            df = Field.read(d, driver=DriverESMFUnstruct)
            self.assertLessEqual(sf.grid.shape[0] - df.grid.shape[0], 150)
            self.assertGreater(sf.grid.shape[0], df.grid.shape[0])

            wgt = os.path.join(chunk_wd, 'esmf_weights_{}.nc'.format(ctr))
            f = Field.read(wgt)
            S = f['S'].v()
            self.assertAlmostEqual(S.min(), 1.0)
            self.assertAlmostEqual(S.max(), 1.0)

        with vm.scoped('merge weights', [0]):
            if not vm.is_null:
                merged_weights = self.get_temporary_file_path('merged_weights.nc')
                gc.create_merged_weight_file(merged_weights, strict=False)
                f = Field.read(merged_weights)
                S = f['S'].v()
                self.assertAlmostEqual(S.min(), 1.0)
                self.assertAlmostEqual(S.max(), 1.0)

    def test_create_dimension_map(self):
        f = self.fixture_esmf_unstruct_field()
        for target in [Topology.POINT, Topology.POLYGON]:
            topo = f.dimension_map.get_topology(target)
            self.assertEqual(topo.get_variable(DMK.X), topo.get_variable(DMK.Y))
        f.grid.parent.load()
        for target in [Topology.POINT, Topology.POLYGON]:
            topo = f.dimension_map.get_topology(target)
            self.assertNotEqual(topo.get_variable(DMK.X), topo.get_variable(DMK.Y))
        self.assertIn(Topology.POINT, f.grid.abstractions_available)

        # Test something in the spatial mask
        self.assertTrue(f.grid.has_mask)
        self.assertIsNotNone(f.dimension_map.get_spatial_mask())

        path = self.get_temporary_file_path('foo.nc')
        f.write(path)
        af = Field.read(uri=path, driver=DriverESMFUnstruct)
        af.grid.parent.load()
        self.assertTrue(af.grid.has_mask)

    def test_create_dimension_map_start_index(self):
        """Test the start index is appropriate set."""

        f = self.fixture_esmf_unstruct_field()
        dmap_start_index = f.dimension_map.get_topology(Topology.POLYGON).get_attrs(DMK.ELEMENT_NODE_CONNECTIVITY)[
            AttributeName.START_INDEX]
        self.assertEqual(dmap_start_index, 1)
        self.assertIsInstance(f.grid, GridUnstruct)
        dmap_start_index = f.dimension_map.get_topology(Topology.POLYGON).get_attrs(DMK.ELEMENT_NODE_CONNECTIVITY)[
            AttributeName.START_INDEX]
        self.assertEqual(dmap_start_index, 1)
        f.grid.abstraction = Topology.POLYGON
        self.assertEqual(f.grid.start_index, 1)

    def test_get_field_write_target(self):
        p1 = 'Polygon ((-116.94238466549290933 52.12861711455555991, -82.00526805089285176 61.59075286434307372, ' \
             '-59.92695130138864101 31.0207758265680269, -107.72286778108455962 22.0438778075388484, ' \
             '-122.76523743459291893 37.08624746104720771, -116.94238466549290933 52.12861711455555991))'
        p2 = 'Polygon ((-63.08099655131782413 21.31602121140134898, -42.70101185946779765 9.42769680782217279, ' \
             '-65.99242293586783603 9.912934538580501, -63.08099655131782413 21.31602121140134898))'
        p1 = wkt.loads(p1)
        p2 = wkt.loads(p2)

        mp1 = MultiPolygon([p1, p2])
        mp2 = mp1.buffer(0.1)
        geoms = [mp1, mp2]
        gvar = GeometryVariable(name='gc', value=geoms, dimensions='elementCount')
        gc = gvar.convert_to(node_dim_name='n_node')
        field = gc.parent
        self.assertEqual(field.grid.node_dim.name, 'n_node')

        actual = DriverESMFUnstruct._get_field_write_target_(field)
        self.assertEqual(field.grid.node_dim.name, 'n_node')
        self.assertNotEqual(id(field), id(actual))
        self.assertEqual(actual['numElementConn'].dtype, np.int32)
        self.assertEqual(actual['elementConn'].dtype, np.int32)
        self.assertNotIn(field.grid.cindex.name, actual)
        self.assertEqual(actual['nodeCoords'].dimensions[0].name, 'nodeCount')

        path = self.get_temporary_file_path('foo.nc')
        actual.write(path)

        # Optional test for loading the mesh file if ESMF is available.
        try:
            import ESMF
        except ImportError:
            pass
        else:
            _ = ESMF.Mesh(filename=path, filetype=ESMF.FileFormat.ESMFMESH)

        path2 = self.get_temporary_file_path('foo2.nc')
        driver = DriverKey.NETCDF_ESMF_UNSTRUCT
        field.write(path2, driver=driver)

        # Test the polygons are equivalent when read from the ESMF unstructured file.
        rd = ocgis.RequestDataset(path2, driver=driver)
        self.assertEqual(rd.driver.key, driver)
        efield = rd.get()
        self.assertEqual(efield.driver.key, driver)
        grid_actual = efield.grid
        self.assertEqual(efield.driver.key, driver)
        self.assertEqual(grid_actual.parent.driver.key, driver)
        self.assertEqual(grid_actual.x.ndim, 1)

        for g in grid_actual.archetype.iter_geometries():
            self.assertPolygonSimilar(g[1], geoms[g[0]])

        ngv = grid_actual.archetype.convert_to()
        self.assertIsInstance(ngv, GeometryVariable)
        # path3 = self.get_temporary_file_path('multis.shp')
        # ngv.write_vector(path3)

    def test_reduce_global(self):
        f = self.fixture_esmf_unstruct_field()
        self.assertIsNotNone(f.grid.cindex)
        self.assertEqual(f.grid.start_index, 1)
        dim = 'elementCount'
        fsub = f[{dim: slice(100, 223)}]
        cindex = fsub.grid.cindex.v()
        offset = 100
        for ii in range(cindex.shape[0]):
            val = [offset + ii - 1 + kk for kk in range(cindex.shape[1])]
            cindex[ii, :] = val
        new_grid = fsub.grid.reduce_global()
        self.assertEqual(new_grid.cindex.v()[0, :].sum(), 10)
        self.assertEqual(new_grid.parent.grid.cindex.v()[0, :].sum(), 10)
