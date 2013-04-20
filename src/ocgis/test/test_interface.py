import unittest
import netCDF4 as nc
from ocgis.interface.interface import GlobalInterface
from ocgis.test.base import TestBase
from ocgis.interface.nc import NcRowDimension
import numpy as np


class TestNcInterface(TestBase):
    
    def test_row_dimension(self):
        value = np.arange(38,40,step=0.5)
#        bounds = np.array([value+0.25,value-0.25]).reshape(-1,2)
        bounds = np.empty((value.shape[0],2))
        bounds[:,0] = value + 0.25
        bounds[:,1] = value - 0.25
        ri = NcRowDimension(value=value)
        self.assertTrue(np.all(value == ri.value))
        self.assertEqual(ri.bounds,None)
        ri = NcRowDimension(value=value,bounds=bounds)
        self.assertTrue(np.all(bounds == ri.bounds))
        with self.assertRaises(ValueError):
            NcRowDimension(bounds=bounds)
        
        ri = NcRowDimension(value=value)
        self.assertEqual(ri.extent,(value.min(),value.max()))
        ri = NcRowDimension(value=value,bounds=bounds)
        self.assertEqual(ri.extent,(bounds.min(),bounds.max()))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()