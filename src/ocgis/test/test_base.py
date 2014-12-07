import datetime
from netCDF4 import Dataset
import os

import numpy as np

from ocgis.interface.base.field import Field
from ocgis.test.base import TestBase


class TestTestBase(TestBase):

    def test_assertNcEqual(self):

        def _write_(fn):
            path = os.path.join(self.current_dir_output, fn)
            with self.nc_scope(path, 'w') as ds:
                var = ds.createVariable('crs', 'c')
                var.name = 'something'
            return path

        path1 = _write_('foo1.nc')
        path2 = _write_('foo2.nc')
        self.assertNcEqual(path1, path2)

    def test_assertNumpyAll_bad_mask(self):
        arr = np.ma.array([1,2,3],mask=[True,False,True])
        arr2 = np.ma.array([1,2,3],mask=[False,True,False])
        with self.assertRaises(AssertionError):
            self.assertNumpyAll(arr,arr2)

    def test_assertNumpyAll_type_differs(self):
        arr = np.ma.array([1,2,3],mask=[True,False,True])
        arr2 = np.array([1,2,3])
        with self.assertRaises(AssertionError):
            self.assertNumpyAll(arr,arr2)

    def test_get_field(self):
        field = self.get_field()
        self.assertIsInstance(field, Field)

    def test_get_time_series(self):
        start = datetime.datetime(1900, 1, 1)
        end = datetime.datetime(1902, 12, 31)
        ret = self.get_time_series(start, end)
        self.assertEqual(ret[0], start)
        self.assertEqual(ret[-1], end)
        self.assertEqual(ret[1]-ret[0], datetime.timedelta(days=1))
        
    def test_multifile(self):
        rd = self.test_data.get_rd('narccap_pr_wrfg_ncep')
        self.assertEqual(len(rd.uri),2)

    def test_ncscope(self):
        path = os.path.join(self.current_dir_output, 'foo.nc')
        with self.nc_scope(path, 'w') as ds:
            self.assertIsInstance(ds, Dataset)


class TestTestData(TestBase):

    def test_get_relative_path(self):
        ret = self.test_data.get_relative_dir('clt_month_units')
        self.assertEqual(ret, 'nc/misc/month_in_time_units')

    def test_size(self):
        size = self.test_data.size
        self.assertGreater(size, 1138333)
