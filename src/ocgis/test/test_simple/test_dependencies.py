import os

import numpy as np
from netCDF4._netCDF4 import MFDataset, MFTime

from ocgis import CoordinateReferenceSystem
from ocgis.test.base import TestBase, attr


@attr('simple')
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

    def test_netCDF4_MFTime(self):
        value = [0, 1, 2]
        units = ['days since 2000-1-1', 'days since 2001-1-1']
        names = ['time_2000.nc', 'time_0001.nc']
        paths = []

        for unit, name in zip(units, names):
            path = self.get_temporary_file_path(name)
            paths.append(path)

            with self.nc_scope(path, 'w', format='NETCDF4_CLASSIC') as f:
                f.createDimension('time')
                vtime = f.createVariable('time', np.float32, dimensions=('time',))
                vtime[:] = value
                vtime.calendar = 'standard'
                vtime.units = unit

        mfd = MFDataset(paths)
        try:
            mvtime = MFTime(mfd.variables['time'])
            desired = [0., 1., 2., 366., 367., 368.]
            actual = mvtime[:].tolist()
            self.assertEqual(actual, desired)
        finally:
            mfd.close()

    def test_osr(self):
        crs = CoordinateReferenceSystem(epsg=4326)
        self.assertNotEqual(crs.value, {})
