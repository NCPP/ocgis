import os
from ocgis import CoordinateReferenceSystem
from ocgis.test.base import TestBase


class TestDependencies(TestBase):

    def test_netCDF4(self):
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('a', 1)
            ds.createDimension('b', 1)
            var = ds.createVariable('foo', int, dimensions=('a', 'b'))
            var[:] = 5
        with self.nc_scope(path) as ds:
            var = ds.variables['foo']
            self.assertEqual(var.shape, (1, 1))

    def test_osr(self):
        crs = CoordinateReferenceSystem(epsg=4326)
        self.assertNotEqual(crs.value, {})
