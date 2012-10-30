import unittest
import netCDF4 as nc
from ocgis.meta.interface.interface import GlobalInterface


class TestInterface(unittest.TestCase):
    
    uri = '/usr/local/climate_data/CanCM4/tasmax_day_CanCM4_decadal2000_r2i1p1_20010101-20101231.nc'
    var_name = 'albisccp'

    
    def test_constructor(self):
        ds = nc.Dataset(self.uri,'r')
        iface = GlobalInterface(ds)
#        level = LevelInterface(ds)
#        ocgds = OcgDataset(self.uri)
        import ipdb;ipdb.set_trace()


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()