import unittest
import netCDF4 as nc
from ocgis.interface.interface import GlobalInterface


class TestInterface(unittest.TestCase):
    dataset = {'uri':'/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc','variable':'tasmax'}

    def setUp(self):
        self.rootgrp = nc.Dataset(self.dataset['uri'])

    def tearDown(self):
        self.rootgrp.close()

    def test_interface(self):
        s_abstractions = ['polygon','point']
        for s_abstraction in s_abstractions:
            overload = {'s_abstraction':s_abstraction}
            i = GlobalInterface(self.rootgrp,self.dataset['variable'],overload=overload)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()