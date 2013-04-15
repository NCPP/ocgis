import unittest
import netCDF4 as nc
from ocgis.interface.interface import GlobalInterface
from ocgis.test.base import TestBase


class TestInterface(TestBase):
    
    def setUp(self):
        self.dataset = self.test_data.get_rd('cancm4_tasmax_2001')
        self.rootgrp = nc.Dataset(self.dataset['uri'])

    def tearDown(self):
        self.rootgrp.close()

    def test_interface(self):
        s_abstractions = ['polygon','point']
        for s_abstraction in s_abstractions:
            overload = {'s_abstraction':s_abstraction}
            GlobalInterface(self.rootgrp,self.dataset['variable'],overload=overload)


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()