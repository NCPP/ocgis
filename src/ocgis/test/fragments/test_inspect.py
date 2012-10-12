import unittest
from ocg.util.inspect import Inspect


class TestInspect(unittest.TestCase):


    def test(self):
        uri = ('/home/local/WX/ben.koziol/Dropbox/nesii/project/ocg/python/'
               'OpenClimateGIS/bin/climate_data/cmip5/albisccp_cfDay_CCSM4_1'
               'pctCO2_r2i1p1_00200101-00391231.nc')
        
        ip = Inspect(uri)
#        t = ip.get_report()
        
        print ip


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()