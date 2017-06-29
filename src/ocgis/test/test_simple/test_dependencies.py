import os
from unittest.case import SkipTest

import numpy as np
from netCDF4._netCDF4 import MFDataset, MFTime

from ocgis import CoordinateReferenceSystem, Variable, vm
from ocgis.test.base import TestBase, attr


@attr('simple')
class TestDependencies(TestBase):
    @attr('mpi', 'optional')
    def test_mpi4py_reduce(self):
        if vm.size != 2:
            raise SkipTest('vm.size != 2')

        if vm.rank == 0:
            n = 2
        else:
            n = 3

        from mpi4py import MPI
        actual = vm.comm.reduce(n, op=MPI.SUM)
        if vm.rank == 0:
            self.assertEqual(actual, 5)

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
        paths = create_mftime_nc_files(self)
        mfd = MFDataset(paths)
        try:
            mvtime = MFTime(mfd.variables['time'])
            desired = [0., 1., 2., 366., 367., 368.]
            actual = mvtime[:].tolist()
            self.assertEqual(actual, desired)
        finally:
            mfd.close()

        # Test units/no calendar on time bounds breaks MFTime.
        paths = create_mftime_nc_files(self, units_on_time_bounds=True)
        mfd = MFDataset(paths)
        try:
            with self.assertRaises(ValueError):
                MFTime(mfd['time_bounds'])
        finally:
            mfd.close()

    def test_osr(self):
        crs = CoordinateReferenceSystem(epsg=4326)
        self.assertNotEqual(crs.value, {})


def create_mftime_nc_files(test_obj, with_all_cf=False, units_on_time_bounds=False, calendar_on_second=True):
    value = [0, 1, 2]
    units = ['days since 2000-1-1', 'days since 2001-1-1']
    names = ['time_2000.nc', 'time_2001.nc']
    paths = []

    if units_on_time_bounds:
        tv = Variable(name='time_bounds_maker', value=value, dimensions='tbmdim')
        tv.set_extrapolated_bounds('time_bounds', 'bounds')

    for ctr, (unit, name) in enumerate(zip(units, names)):
        path = test_obj.get_temporary_file_path(name)
        paths.append(path)

        with test_obj.nc_scope(path, 'w', format='NETCDF4_CLASSIC') as f:
            f.createDimension('time')
            vtime = f.createVariable('time', np.float32, dimensions=('time',))
            vtime[:] = value
            vtime.units = unit
            if ctr == 0 or (calendar_on_second and ctr == 1):
                vtime.calendar = 'standard'
            vtime.axis = 'T'
            if units_on_time_bounds:
                vtime.calendar = 'noleap'
                vtime.bounds = 'time_bounds'

            if units_on_time_bounds:
                f.createDimension('bounds', 2)
                vbtime = f.createVariable('time_bounds', np.float32, dimensions=('time', 'bounds'))
                vbtime[:] = tv.bounds.get_value()
                # Note no calendar on bounds variable.
                vbtime.units = units

            if with_all_cf:
                f.createDimension('x', 4)
                f.createDimension('y', 5)
                x = f.createVariable('x', float, dimensions=('x',))
                x[:] = [100., 101., 102., 103.]
                x.axis = 'X'
                y = f.createVariable('y', float, dimensions=('y',))
                y[:] = [40., 41., 42., 43., 44.]
                y.axis = 'Y'
                data = f.createVariable('data', float, dimensions=('time', 'y', 'x'))
                data[:] = 1

    return paths
