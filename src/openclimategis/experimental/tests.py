import unittest
from shapely.geometry.polygon import Polygon
from util.ncwrite import NcSpatial, NcTime, NcVariable, NcWrite
import datetime
import os
from netCDF4 import Dataset


class TestSimpleNc(unittest.TestCase):

    def get_uri(self,constant=5,nlevels=1):
        """
        constant=5 -- provides a constant value when generating data to fill
            a NC variable. set to None to generate random values.
        nlevels=1 -- a value greater than 1 will create a NC with a fourth level
            dimension. the input values determines the number of levels.
        """
        bounds = Polygon(((0,0),(10,0),(10,15),(0,15)))
        res = 5
        ncspatial = NcSpatial(bounds,res)
        
        rng = [datetime.datetime(2007,10,1),datetime.datetime(2007,10,3)]
        interval = datetime.timedelta(days=1)
        nctime = NcTime(rng,interval)
        
        ncvariable = NcVariable("Prcp","mm",constant=constant)
        
        ncw = NcWrite(ncvariable,ncspatial,nctime,nlevels=nlevels)
        uri = ncw.write()
        return(uri)

    def test_get_uri(self):
        uri = self.get_uri()
        self.assertTrue(os.path.exists(uri))
        d = Dataset(uri,'r')
        sh = d.variables["Prcp"].shape
        self.assertEqual(sh,(3,4,3))
        d.close()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()