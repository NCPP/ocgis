import itertools

from shapely import wkt
from shapely.geometry import MultiPolygon

import ocgis
from ocgis import GeometryVariable, vm, Field
from ocgis.base import get_variable_names
from ocgis.constants import DriverKey, VariableName
from ocgis.driver.nc_esmf_unstruct import DriverESMFUnstruct
from ocgis.driver.request.core import RequestDataset
from ocgis.spatial.geomc import PolygonGC
from ocgis.spatial.grid import GridUnstruct
from ocgis.test.base import TestBase, attr
from ocgis.variable.crs import WGS84, Spherical, create_crs


class TestDriverESMFUnstruct(TestBase):
    @attr('mpi', 'slow')
    def test_system_converting_state_boundaries_shapefile(self):
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

            rd = RequestDataset(uri=self.path_state_boundaries)
            rd.metadata['schema']['geometry'] = 'MultiPolygon'
            field = rd.get()

            # Test there is no mask present.
            field.geom.load()
            self.assertFalse(field.geom.has_mask)
            self.assertNotIn(VariableName.SPATIAL_MASK, field)
            self.assertIsNone(field.dimension_map.get_spatial_mask())

            self.assertEqual(field.crs, WGS84())
            if k.transform_to_crs is not None:
                field.update_crs(desired_crs)
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

            actual_xsums.append(gc.x.get_value().sum())
            actual_ysums.append(gc.y.get_value().sum())
            self.assertEqual(gc.crs, desired_crs)

            # Test there is no mask present after conversion to geometry coordinates.
            self.assertFalse(gc.has_mask)
            self.assertNotIn(VariableName.SPATIAL_MASK, gc.parent)
            self.assertIsNone(gc.dimension_map.get_spatial_mask())

            for v in list(field.values()):
                if v.name != field.geom.name:
                    gc.parent.add_variable(v.extract(), force=True)

            path = self.get_temporary_file_path('esmf_state_boundaries.nc')
            self.assertEqual(gc.parent.crs, desired_crs)
            gc.parent.write(path, driver=DriverKey.NETCDF_ESMF_UNSTRUCT)

            gathered_geoms = vm.gather(field.geom.get_value())
            if vm.rank == 0:
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

        vm.barrier()

        # Test coordinates have actually changed.
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

        # Test the new object does not share data with the source.
        for dn in data_variable_names:
            self.assertNotIn(dn, gc.parent)

        self.assertFalse(sub.geom.has_allocated_value)
        self.assertTrue(field.geom.protected)
        path = self.get_temporary_file_path('out.nc')
        gc.parent.write(path)

    def test_get_field_write_target(self):
        p1 = 'Polygon ((-116.94238466549290933 52.12861711455555991, -82.00526805089285176 61.59075286434307372, -59.92695130138864101 31.0207758265680269, -107.72286778108455962 22.0438778075388484, -122.76523743459291893 37.08624746104720771, -116.94238466549290933 52.12861711455555991))'
        p2 = 'Polygon ((-63.08099655131782413 21.31602121140134898, -42.70101185946779765 9.42769680782217279, -65.99242293586783603 9.912934538580501, -63.08099655131782413 21.31602121140134898))'
        p1 = wkt.loads(p1)
        p2 = wkt.loads(p2)

        mp1 = MultiPolygon([p1, p2])
        mp2 = mp1.buffer(0.1)
        geoms = [mp1, mp2]
        gvar = GeometryVariable(name='gc', value=geoms, dimensions='elementCount')
        gc = gvar.convert_to(node_dim_name='nodeCount')
        field = gc.parent

        actual = DriverESMFUnstruct._get_field_write_target_(field)
        self.assertNotEqual(id(field), id(actual))

        path = self.get_temporary_file_path('foo.nc')
        actual.write(path)

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
