import unittest
from netCDF4 import Dataset
from ocg.meta.interface.projection import get_projection


class TestProjection(unittest.TestCase):
    uri = ('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/'
           'OpenClimateGIS/bin/climate_data/hostetler/'
           'RegCM3_Daily_srm_GFDL.ncml.nc')
    
    def test(self):
        ds = Dataset(self.uri)
        proj = get_projection(ds)
        import ipdb;ipdb.set_trace()

if __name__ == "__main__":
#    import sys;sys.argv = ['', 'TestSimpleMultiCalc01.test']
    unittest.main()